import torch

class MyDataset:
    def __init__(self) -> None:
        self.enviroment = []
        self.labels = []
        self.latest_enviroment = []
        self.latest_labels = []

    def add(self, enviroment, labels):
        self.enviroment.append(enviroment)
        self.labels.append(labels)
        self.latest_enviroment = enviroment
        self.latest_labels = labels

    def get_batch(self, batch_size=1):
        indices = torch.randint(0, len(self.enviroment), (batch_size,))
        return self.enviroment[indices], self.labels[indices]


class RolloutBuffer:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((num_steps, num_envs, 2), dtype=torch.long)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.advantages = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.returns = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.step = 0

    def add(self, obs, actions, logprobs, rewards, dones, values):
        self.obs[self.step].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.logprobs[self.step].copy_(logprobs)
        self.rewards[self.step].copy_(rewards)
        self.dones[self.step].copy_(dones)
        self.values[self.step].copy_(values)
        self.step += 1

    def compute_gae(self, last_values: torch.Tensor, gamma: float, gae_lambda: float):
        next_advantage = torch.zeros(self.num_envs, dtype=torch.float32)
        next_value = last_values
        T = int(self.step)
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
            self.advantages[t] = next_advantage
            next_value = self.values[t]
        self.returns[:T] = self.advantages[:T] + self.values[:T]
