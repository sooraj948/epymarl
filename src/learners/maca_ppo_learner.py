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
        if beta > 0:
            from modules.maca_adapter import MACAAdapter
            self.adapter = MACAAdapter(args, scheme, self.critic, logger)

    def train(self, batch, t_env, episode_num):
        if self.adapter is None:
            return super().train(batch, t_env, episode_num)
        start = time.perf_counter()
        self.shaped_rewards = self.adapter.prepare(batch, t_env)
        prepared = time.perf_counter()
        try:
            result = super().train(batch, t_env, episode_num)
        finally:
            self.shaped_rewards = None
        self.logger.log_stat("maca/prepare_seconds", prepared - start, t_env)
        self.logger.log_stat("maca/learner_seconds", time.perf_counter() - prepared, t_env)
        return result

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        if self.shaped_rewards is not None:
            rewards = self.shaped_rewards
        return super().train_critic_sequential(critic, target_critic, batch, rewards, mask)

    def save_models(self, path):
        super().save_models(path)
        torch.save({
            "adapter": self.adapter.state_dict() if self.adapter is not None else None,
            "target_critic": self.target_critic.state_dict(),
            "critic_training_steps": self.critic_training_steps,
            "last_target_update_step": self.last_target_update_step,
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
