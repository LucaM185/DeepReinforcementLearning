import torch
import torch.nn as nn
import torch.nn.functional as F

# Exposed defaults for easy tuning
ACTOR_DEFAULT_HIDDEN_SIZE = 128
ACTOR_DEFAULT_N_LAYERS = 2
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_COEF = 0.2
PPO_UPDATE_EPOCHS = 4
PPO_MINIBATCH_SIZE = 256
PPO_ENTROPY_COEF = 0.01
PPO_VALUE_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5

# ==========================
# PPO: Actor-Critic + Trainer
# ==========================

class ActorCritic(nn.Module):
    def __init__(self, in_size: int, hidden_size: int = ACTOR_DEFAULT_HIDDEN_SIZE, n_layers: int = ACTOR_DEFAULT_N_LAYERS):
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

    def get_action_and_value(self, obs: torch.Tensor, actions: torch.Tensor = None, lateral_logits_bias: torch.Tensor = None):
        logits_lat, logits_long, value = self.forward(obs)
        # Optional exploration bias on lateral logits (shape [3] or [batch,3])
        if lateral_logits_bias is not None:
            if lateral_logits_bias.dim() == 1:
                logits_lat = logits_lat + lateral_logits_bias.view(1, -1)
            else:
                logits_lat = logits_lat + lateral_logits_bias
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
        lr: float = PPO_LR,
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_GAE_LAMBDA,
        clip_coef: float = PPO_CLIP_COEF,
        update_epochs: int = PPO_UPDATE_EPOCHS,
        minibatch_size: int = PPO_MINIBATCH_SIZE,
        entropy_coef: float = PPO_ENTROPY_COEF,
        value_coef: float = PPO_VALUE_COEF,
        max_grad_norm: float = PPO_MAX_GRAD_NORM,
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
