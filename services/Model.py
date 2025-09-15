import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    setup = [26, 6, 128, 2]
    
    def __init__(self, in_size, out_size, hidden_size, n_layers, lr=0.003):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fcx = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]) # this is a list of linear layers
        self.fc2 = nn.Linear(hidden_size, out_size)
        self.lnorm = nn.LayerNorm(hidden_size)
        self.lr = lr
    
    def forward(self, inputs):
        x = F.gelu(self.fc1(inputs))
        for hidden in self.fcx: # iterating over hidden layers
            x = self.lnorm(F.gelu(hidden(x)))  # applying each hidden layer
        return torch.softmax(self.fc2(x).view(-1, 2, 3), axis=-1).view(-1, 6)

    def get_model_copy(self):
        state = self.state_dict()
        newmodel = MLP(*MLP.setup, self.lr*0.8)
        newmodel.load_state_dict(state)
        return newmodel

    def train(self, enviroment, actions, past_interaction=50): 
        past_interaction *= 1

        buttons = actions
        radar = enviroment

        # Align lengths in case recordings differ (e.g., steering stops when crashed)
        with torch.no_grad():
            len_env = radar.shape[0] if radar.ndim > 0 else 0
            len_act = buttons.shape[0] if buttons.ndim > 0 else 0
            min_len = min(len_env, len_act)
            if min_len <= 0:
                return self.get_model_copy()
            radar = radar[:min_len]
            buttons = buttons[:min_len]

        # crashed_timestamp = radar.argmin(-1) if (radar[radar.argmin(-1)].sum() == 0) else -1 
        # context = 1000
        # buttons = buttons[crashed_timestamp-context:crashed_timestamp]
        # radar = radar[crashed_timestamp-context:crashed_timestamp]
        thresh = 2.5
        buttons[:-past_interaction][radar[:-past_interaction][:, -1] < thresh] = 1-buttons[:-past_interaction][radar[:-past_interaction][:, -1] < thresh]
        buttons[-past_interaction:] = 1-buttons[-past_interaction:]
        if buttons.shape[0] < 4000:
            buttons = buttons[-past_interaction-250:]
            radar = radar[-past_interaction-250:]

        # print(buttons)

        # train model   
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.8)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        from tqdm import tqdm
        for epoch in (range(10)):
            optimizer.zero_grad()

            output = self(radar)
            loss = F.mse_loss(output, buttons)
            loss.backward()
            optimizer.step()
            # p.set_description(f"Loss: {loss.item():2f} at epoch {epoch:2d}")
        
        state = self.state_dict()
        newmodel = MLP(*MLP.setup, self.lr)
        newmodel.load_state_dict(state)
        return newmodel


# ==========================
# PPO: Actor-Critic + Trainer
# ==========================

class ActorCritic(nn.Module):
    def __init__(self, in_size: int, hidden_size: int = 128, n_layers: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(hidden_size)
        # Actor heads: lateral (3), longitudinal (3)
        self.actor_lat = nn.Linear(hidden_size, 3)
        self.actor_long = nn.Linear(hidden_size, 3)
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)

    def _body(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(obs))
        for layer in self.hidden_layers:
            x = self.ln(F.gelu(layer(x)))
        return x

    def forward(self, obs: torch.Tensor):
        x = self._body(obs)
        logits_lat = self.actor_lat(x)
        logits_long = self.actor_long(x)
        value = self.critic(x).squeeze(-1)
        return logits_lat, logits_long, value

    def get_action_and_value(self, obs: torch.Tensor, actions: torch.Tensor = None):
        logits_lat, logits_long, value = self.forward(obs)
        dist_lat = torch.distributions.Categorical(logits=logits_lat)
        dist_long = torch.distributions.Categorical(logits=logits_long)
        if actions is None:
            action_lat = dist_lat.sample()
            action_long = dist_long.sample()
        else:
            action_lat = actions[:, 0]
            action_long = actions[:, 1]
        logprob = dist_lat.log_prob(action_lat) + dist_long.log_prob(action_long)
        entropy = dist_lat.entropy() + dist_long.entropy()
        actions_out = torch.stack([action_lat, action_long], dim=-1)
        return actions_out, logprob, entropy, value


class PPOTrainer:
    def __init__(
        self,
        model: ActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        update_epochs: int = 4,
        minibatch_size: int = 256,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    def update(self, buffer):
        T = int(buffer.step)
        obs = buffer.obs[:T].reshape(-1, buffer.obs.shape[-1])
        actions = buffer.actions[:T].reshape(-1, 2)
        logprobs = buffer.logprobs[:T].reshape(-1)
        advantages = buffer.advantages[:T].reshape(-1)
        returns = buffer.returns[:T].reshape(-1)
        values = buffer.values[:T].reshape(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        num_samples = obs.shape[0]
        idxs = torch.randperm(num_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.update_epochs):
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = idxs[start:end]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_logprobs = logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                _, new_logprob, entropy, new_values = self.model.get_action_and_value(mb_obs, mb_actions)
                ratio = (new_logprob - mb_logprobs).exp()

                # Policy loss with clipping
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                value_clipped = values[mb_idx] + (new_values - values[mb_idx]).clamp(-self.clip_coef, self.clip_coef)
                value_losses = (new_values - mb_returns).pow(2)
                value_losses_clipped = (value_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss).item()

        updates = max(1, (num_samples + self.minibatch_size - 1) // self.minibatch_size) * self.update_epochs
        return {
            "policy_loss": total_policy_loss / updates,
            "value_loss": total_value_loss / updates,
            "entropy": total_entropy / updates,
        }
