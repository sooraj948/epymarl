import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """Simple MLP Policy Network for REINFORCE."""
    def __init__(self, obs_dim: int, act_dim: int):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

class ReinforceAgent:
    """Independent REINFORCE Agent with Entropy Regularization and Batched Updates."""
    def __init__(self, obs_dim: int, act_dim: int, lr: float = 0.001, gamma: float = 0.99, 
                 entropy_coef: float = 0.01, batch_size: int = 16):
        self.policy_net = PolicyNetwork(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.memory = []        # Current episode memory: stores (log_prob, entropy, reward)
        self.batch_memory = []  # Accumulated batch of episode memories (list of lists)

    def get_action(self, obs) -> tuple[int, torch.Tensor]:
        """Select an action and return the action and its log probability."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        
        logits = self.policy_net(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.last_entropy = dist.entropy()
        
        return action.item(), log_prob

    def store_reward(self, log_prob: torch.Tensor, reward: float):
        """Store the log_prob and the reward for this step."""
        entropy = getattr(self, 'last_entropy', torch.tensor(0.0))
        self.memory.append((log_prob, entropy, reward))

    def clear_memory(self):
        """Reset the current episode memory."""
        self.memory = []

    def update_policy(self, force: bool = False):
        """
        Accumulate the current episode memory and perform policy updates in mini-batches.
        
        If force=True, performs update immediately on any remaining trajectories in batch_memory.
        """
        if self.memory:
            self.batch_memory.append(list(self.memory))
            self.clear_memory()

        if not self.batch_memory:
            return

        # Check if we should execute a policy update step
        if not force and len(self.batch_memory) < self.batch_size:
            return

        all_log_probs = []
        all_entropies = []
        all_returns = []

        # 1. Compute discounted returns for each episode in the batch individually
        for ep_memory in self.batch_memory:
            R = 0
            ep_returns = []
            for (log_prob, entropy, reward) in reversed(ep_memory):
                R = reward + self.gamma * R
                ep_returns.append(R)
            ep_returns.reverse()
            
            all_returns.extend(ep_returns)
            for (log_prob, entropy, _) in ep_memory:
                all_log_probs.append(log_prob)
                all_entropies.append(entropy)

        # 2. Standardize returns across all steps of all episodes in the batch
        returns_tensor = torch.tensor(all_returns, dtype=torch.float32)
        if len(returns_tensor) > 1 and returns_tensor.std() > 0:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-6)
        else:
            returns_tensor = returns_tensor - returns_tensor.mean()

        # 3. Calculate the policy loss
        policy_loss = []
        for log_prob, R in zip(all_log_probs, returns_tensor):
            policy_loss.append(-log_prob * R)

        # 4. Perform Adam gradient step
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() - self.entropy_coef * torch.stack(all_entropies).sum()
        loss.backward()
        self.optimizer.step()

        # 5. Clear the batch memory
        self.batch_memory = []

class ValueNetwork(nn.Module):
    """Simple MLP Value Network for MAPPO Critic."""
    def __init__(self, obs_dim: int):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def update_shared_critic(critic_net, critic_optimizer, critic_obs, returns,
                         value_coef, max_grad_norm, ppo_epochs):
    """Fit one agent-conditioned value function on the pooled agent targets."""
    for _ in range(ppo_epochs):
        values = critic_net(critic_obs)
        value_loss = 0.5 * F.mse_loss(values, returns)
        critic_optimizer.zero_grad()
        (value_coef * value_loss).backward()
        nn.utils.clip_grad_norm_(critic_net.parameters(), max_grad_norm)
        critic_optimizer.step()

class MAPPOAgent:
    """MAPPO Agent with GAE and clipped PPO objective."""
    def __init__(self, obs_dim: int, act_dim: int, critic_obs_dim: int = None,
                 lr: float = 0.001, critic_lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_eps: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5, ppo_epochs: int = 4,
                 shared_critic_net=None, shared_critic_optimizer=None,
                 critic_agent_id=None, n_agents=None):
        if critic_obs_dim is None:
            critic_obs_dim = obs_dim
            
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.critic_obs_dim = critic_obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        self.actor_net = PolicyNetwork(obs_dim, act_dim)
        self.shared_critic = shared_critic_net is not None
        if self.shared_critic:
            if shared_critic_optimizer is None or critic_agent_id is None or n_agents is None:
                raise ValueError("Shared critic requires an optimizer, agent ID, and agent count")
            if not 0 <= critic_agent_id < n_agents:
                raise ValueError("Invalid critic agent ID")
            self.critic_agent_id = critic_agent_id
            self.n_agents = n_agents
        self.critic_net = shared_critic_net if self.shared_critic else ValueNetwork(critic_obs_dim)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=lr)
        self.critic_optimizer = shared_critic_optimizer if self.shared_critic else optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.last_update_diagnostics = {}
        self.last_actor_gradient_norms = []

    def critic_input(self, critic_obs):
        if not self.shared_critic:
            return critic_obs
        agent_id = F.one_hot(
            torch.tensor(self.critic_agent_id, device=critic_obs.device),
            num_classes=self.n_agents,
        ).to(dtype=critic_obs.dtype)
        return torch.cat((critic_obs, agent_id.expand(*critic_obs.shape[:-1], -1)), dim=-1)

    def get_action(self, obs, critic_obs=None) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an action, return action index, action log probability, and estimated value.
        """
        if not isinstance(obs, torch.Tensor):
            obs_t = torch.tensor(obs, dtype=torch.float32)
        else:
            obs_t = obs.float()

        if critic_obs is None:
            c_obs_t = obs_t
        elif not isinstance(critic_obs, torch.Tensor):
            c_obs_t = torch.tensor(critic_obs, dtype=torch.float32)
        else:
            c_obs_t = critic_obs.float()

        logits = self.actor_net(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        value = self.critic_net(self.critic_input(c_obs_t))

        return action.item(), log_prob.detach(), value.detach()

    def compute_gae(self, rewards, values, dones, next_value: float = 0.0):
        """
        Compute Generalized Advantage Estimation (GAE) and target returns.
        """
        T = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32)
        deltas = torch.zeros(T, dtype=torch.float32)

        if isinstance(values, list):
            values_tensor = torch.stack([v.detach().cpu() if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values])
        else:
            values_tensor = values.detach().cpu()

        last_gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_v = next_value
            else:
                next_v = values_tensor[t + 1].item()

            done_float = 1.0 if dones[t] else 0.0
            delta = float(rewards[t]) + self.gamma * next_v * (1.0 - done_float) - values_tensor[t].item()
            deltas[t] = delta
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - done_float) * last_gae
            advantages[t] = float(last_gae)

        returns = advantages + values_tensor
        return advantages, returns, deltas

    def update(self, batch_obs: torch.Tensor, batch_critic_obs: torch.Tensor,
               batch_actions: torch.Tensor, batch_old_log_probs: torch.Tensor,
               batch_advantages: torch.Tensor, batch_returns: torch.Tensor,
               diagnostic_plain_advantages: torch.Tensor = None,
               diagnostic_episode_ids: torch.Tensor = None,
               diagnostic_target_episodes=None, log_gradient_norms=False):
        """
        Perform PPO clipped policy and value updates for ppo_epochs.
        """
        # Advantage standardization over batch
        if len(batch_advantages) > 1 and batch_advantages.std() > 1e-8:
            advs = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
        else:
            advs = batch_advantages - batch_advantages.mean()

        diagnostic_results = {}
        self.last_update_diagnostics = {}
        self.last_actor_gradient_norms = []
        plain_advs = None
        if diagnostic_plain_advantages is not None and diagnostic_episode_ids is not None:
            plain_batch = diagnostic_plain_advantages
            if len(plain_batch) > 1 and plain_batch.std() > 1e-8:
                plain_advs = (plain_batch - plain_batch.mean()) / (plain_batch.std() + 1e-8)
            else:
                plain_advs = plain_batch - plain_batch.mean()

            def policy_gradient(advantages, mask):
                logits = self.actor_net(batch_obs[mask])
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions[mask])
                ratios = torch.exp(log_probs - batch_old_log_probs[mask])
                objective = torch.min(
                    ratios * advantages[mask],
                    torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * advantages[mask],
                ).mean()
                gradients = torch.autograd.grad(
                    objective,
                    tuple(self.actor_net.parameters()),
                    retain_graph=False,
                    allow_unused=True,
                )
                parts = [
                    gradient.reshape(-1)
                    for gradient in gradients
                    if gradient is not None
                ]
                return torch.cat(parts) if parts else torch.zeros(1)

            if diagnostic_target_episodes is None:
                full_mask = torch.ones_like(diagnostic_episode_ids, dtype=torch.bool)
                batch_shaped_gradient = policy_gradient(advs, full_mask)
                batch_plain_gradient = policy_gradient(plain_advs, full_mask)
            else:
                batch_shaped_gradient = batch_plain_gradient = None
            if batch_shaped_gradient is not None:
                batch_shaped_norm = float(
                torch.linalg.vector_norm(batch_shaped_gradient).item()
                )
                batch_plain_norm = float(
                torch.linalg.vector_norm(batch_plain_gradient).item()
                )
            else:
                batch_shaped_norm = batch_plain_norm = float('nan')
            batch_denominator = batch_shaped_norm * batch_plain_norm
            self.last_update_diagnostics = {
                "transition_count": int(len(diagnostic_episode_ids)),
                "episode_count": int(torch.unique(diagnostic_episode_ids).numel()),
                "standardized_adv_sign_flip_fraction": float(
                    (torch.sign(plain_advs) != torch.sign(advs)).float().mean().item()
                ),
                "preupdate_shaped_grad_norm": batch_shaped_norm,
                "preupdate_plain_grad_norm": batch_plain_norm,
                "preupdate_gradient_cosine": (
                    float(
                        torch.dot(batch_shaped_gradient, batch_plain_gradient).item()
                        / batch_denominator
                    )
                    if batch_denominator > 0.0
                    else float("nan")
                ),
                "preupdate_gradient_norm_ratio": (
                    batch_shaped_norm / (batch_plain_norm + 1e-12)
                ),
                "ppo_clip_fraction_by_epoch": [],
                "shaped_surrogate_by_epoch": [],
                "plain_surrogate_by_epoch": [],
            }

            for episode_tensor in torch.unique(diagnostic_episode_ids):
                episode = int(episode_tensor.item())
                if diagnostic_target_episodes is not None and episode not in diagnostic_target_episodes:
                    continue
                mask = diagnostic_episode_ids == episode_tensor
                shaped_gradient = policy_gradient(advs, mask)
                plain_gradient = policy_gradient(plain_advs, mask)
                shaped_norm = float(torch.linalg.vector_norm(shaped_gradient).item())
                plain_norm = float(torch.linalg.vector_norm(plain_gradient).item())
                denominator = shaped_norm * plain_norm
                cosine = (
                    float(torch.dot(shaped_gradient, plain_gradient).item() / denominator)
                    if denominator > 0.0
                    else float("nan")
                )
                diagnostic_results[episode] = {
                    "plain_batch_mean": float(plain_batch.mean().item()),
                    "plain_batch_std": float(plain_batch.std().item()) if len(plain_batch) > 1 else 0.0,
                    "shaped_batch_mean": float(batch_advantages.mean().item()),
                    "shaped_batch_std": float(batch_advantages.std().item()) if len(batch_advantages) > 1 else 0.0,
                    "standardized_plain_adv_mean": float(plain_advs[mask].mean().item()),
                    "standardized_shaped_adv_mean": float(advs[mask].mean().item()),
                    "standardized_adv_sign_flip_fraction": float(
                        (torch.sign(plain_advs[mask]) != torch.sign(advs[mask]))
                        .float()
                        .mean()
                        .item()
                    ),
                    "ppo_clip_fraction_by_epoch": [],
                    "shaped_surrogate_by_epoch": [],
                    "plain_surrogate_by_epoch": [],
                    "preupdate_shaped_grad_norm": shaped_norm,
                    "preupdate_plain_grad_norm": plain_norm,
                    "preupdate_gradient_cosine": cosine,
                }

        for epoch in range(self.ppo_epochs):
            # Forward pass through actor & critic
            logits = self.actor_net(batch_obs)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            if not self.shared_critic:
                new_values = self.critic_net(batch_critic_obs)

            # Ratio for PPO
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advs
            policy_loss = -torch.min(surr1, surr2).mean()

            if plain_advs is not None:
                with torch.no_grad():
                    plain_surr = torch.min(
                        ratio * plain_advs,
                        torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                        * plain_advs,
                    )
                    clipped = (ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)
                    shaped_surr = torch.min(surr1, surr2)
                    self.last_update_diagnostics["ppo_clip_fraction_by_epoch"].append(
                        float(clipped.float().mean().item())
                    )
                    self.last_update_diagnostics["shaped_surrogate_by_epoch"].append(
                        float(shaped_surr.mean().item())
                    )
                    self.last_update_diagnostics["plain_surrogate_by_epoch"].append(
                        float(plain_surr.mean().item())
                    )
                    for episode in diagnostic_results:
                        mask = diagnostic_episode_ids == episode
                        diagnostic_results[episode]["ppo_clip_fraction_by_epoch"].append(
                            float(clipped[mask].float().mean().item())
                        )
                        diagnostic_results[episode]["shaped_surrogate_by_epoch"].append(
                            float(shaped_surr[mask].mean().item())
                        )
                        diagnostic_results[episode]["plain_surrogate_by_epoch"].append(
                            float(plain_surr[mask].mean().item())
                        )

            # Value loss
            if not self.shared_critic:
                value_loss = 0.5 * nn.functional.mse_loss(new_values, batch_returns)

            # Total actor loss
            actor_loss = policy_loss - self.entropy_coef * entropy

            # Optimize Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            if log_gradient_norms:
                self.last_actor_gradient_norms.append({
                    "ppo_epoch": epoch + 1,
                    "pre_clip_grad_norm": float(actor_grad_norm.item()),
                })
            self.actor_optimizer.step()

            # Optimize Critic
            if not self.shared_critic:
                self.critic_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        return diagnostic_results
