"""MACA reward preparation with the unchanged EPyMARL n-step PPO update."""

from pathlib import Path
import time

import torch

from .ppo_learner import PPOLearner


class MACAPPOLearner(PPOLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        beta = args.maca["beta"]
        if beta < 0 or not torch.isfinite(torch.tensor(beta)):
            raise ValueError("MACA beta must be finite and nonnegative")
        self.adapter = None
        self.shaped_rewards = None
        self._preparation_seconds = 0.0
        if beta > 0:
            from modules.maca_adapter import MACAAdapter
            self.adapter = MACAAdapter(args, scheme, self.critic, logger)

    def train(self, batch, t_env, episode_num):
        if self.adapter is None:
            return super().train(batch, t_env, episode_num)
        start = time.perf_counter()
        self._preparation_seconds = 0.0
        self._t_env = t_env
        try:
            result = super().train(batch, t_env, episode_num)
        finally:
            self.shaped_rewards = None
        self.logger.log_stat("maca/prepare_seconds", self._preparation_seconds, t_env)
        self.logger.log_stat("maca/learner_seconds", time.perf_counter() - start - self._preparation_seconds, t_env)
        return result

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        if self.adapter is not None:
            if self.shaped_rewards is None:
                # The base learner has updated environment-only reward moments once.
                # Prepare here, before any critic update, and cache for all PPO epochs.
                start = time.perf_counter()
                if self.args.standardise_rewards:
                    mean, std = self.rew_ms.mean, torch.sqrt(self.rew_ms.var)
                    self.adapter.pipeline.frozen.set_reward_normalization(mean.item(), std.item())
                self.shaped_rewards = self.adapter.prepare(batch, self._t_env)
                if self.args.standardise_rewards:
                    self.shaped_rewards = (self.shaped_rewards - mean) / std
                self._preparation_seconds = time.perf_counter() - start
            rewards = self.shaped_rewards
        return super().train_critic_sequential(critic, target_critic, batch, rewards, mask)

    def save_models(self, path):
        super().save_models(path)
        torch.save({
            "adapter": self.adapter.state_dict() if self.adapter is not None else None,
            "target_critic": self.target_critic.state_dict(),
            "critic_training_steps": self.critic_training_steps,
            "last_target_update_step": self.last_target_update_step,
            "reward_normalization": {
                "mean": self.rew_ms.mean, "var": self.rew_ms.var, "count": self.rew_ms.count,
            } if self.args.standardise_rewards else None,
        }, Path(path) / "maca_state.th")

    def load_models(self, path):
        super().load_models(path)
        self.old_mac.load_state(self.mac)
        state = torch.load(Path(path) / "maca_state.th", map_location="cpu", weights_only=False)
        if (state["adapter"] is None) != (self.adapter is None):
            raise ValueError("MACA checkpoint beta mode does not match this run")
        if self.adapter is not None:
            self.adapter.load_state_dict(state["adapter"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.critic_training_steps = state["critic_training_steps"]
        self.last_target_update_step = state["last_target_update_step"]
        moments = state.get("reward_normalization")
        if self.args.standardise_rewards:
            if moments is None:
                raise ValueError("Standardized MACA checkpoint is missing reward moments")
            for name, value in moments.items():
                setattr(self.rew_ms, name, value.to(self.rew_ms.mean.device) if torch.is_tensor(value) else value)
