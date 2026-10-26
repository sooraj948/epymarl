"""Stock QMIX optimization with optional fixed-episode MACA objectives."""

import copy
from pathlib import Path

import numpy as np
import torch as th

from learners.q_learner import QLearner
from modules.qmix_maca_adapter import QMIXMACAAdapter


def terminal_credit_rewards(rewards, credit, lengths):
    """Broadcast a common stream to agents and add each bonus exactly once."""
    shaped = rewards.expand(-1, -1, credit.shape[-1]).clone()
    rows = th.arange(shaped.shape[0], device=shaped.device)
    shaped[rows, lengths.long() - 1] += credit
    return shaped


def finite_episode_returns(rewards, mask, gamma):
    """Monte Carlo targets for complete finite episodes, including timeouts."""
    targets = th.zeros_like(rewards)
    running = th.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        running = (rewards[:, t] + gamma * running) * mask[:, t]
        targets[:, t] = running
    return targets


class LBFQMIXLearner(QLearner):
    @staticmethod
    def extend_scheme(args, scheme):
        if hasattr(args, "maca"):
            scheme["maca_credit"] = {"vshape": (args.n_agents,), "episode_const": True}

    def __init__(self, mac, scheme, logger, args):
        if args.mixer != "qmix" or not args.common_reward or args.standardise_returns:
            raise ValueError("LBF QMIX requires common-reward QMIX without return standardization")
        settings = getattr(args, "maca", None)
        if settings is not None:
            if (settings.get("mode") not in {"individual_aux", "team_sum"}
                    or not np.isfinite(settings.get("beta", .1)) or settings.get("beta", .1) < 0
                    or not np.isfinite(settings.get("auxiliary_weight", 1.))
                    or settings.get("auxiliary_weight", 1.) < 0):
                raise ValueError("Invalid QMIX MACA mode, beta or auxiliary weight")
        super().__init__(mac, scheme, logger, args)
        self.adapter = None
        self.mode = None if settings is None else settings["mode"]
        self.auxiliary_weight = 0. if settings is None else settings.get("auxiliary_weight", 1.)
        if settings is not None and settings.get("beta", .1) > 0:
            self.adapter = QMIXMACAAdapter(
                args, scheme, (self.target_mac, self.target_mixer), logger,
            )

    def prepare_episode(self, batch, t_env):
        if "maca_credit" not in batch.scheme:
            return
        credit = (batch["reward"].new_zeros(batch.batch_size, self.n_agents) if self.adapter is None
                  else self.adapter.prepare(batch, t_env, self.last_target_update_step))
        batch.update({"maca_credit": credit}, mark_filled=False)

    def _credit_in_training_units(self, batch):
        credit = batch["maca_credit"].detach()
        if self.args.standardise_rewards:
            credit = credit / th.sqrt(self.rew_ms.var)
        return credit

    def _prepare_rewards(self, batch, rewards, mask, t_env):
        # Match the stock environment-only replay normalizer, including its
        # padding convention. Credit is added after updating those moments.
        rewards = super()._prepare_rewards(batch, rewards, mask, t_env)
        if self.adapter is not None and self.mode == "team_sum":
            credit = self._credit_in_training_units(batch).sum(-1, keepdim=True)
            lengths = batch["filled"].sum(1).squeeze(-1) - 1
            rewards = terminal_credit_rewards(rewards, credit, lengths)
        return rewards

    def _auxiliary_loss(self, batch, chosen_agent_qvals, rewards, mask, t_env):
        if self.adapter is None or self.mode != "individual_aux" or self.auxiliary_weight == 0:
            return chosen_agent_qvals.new_zeros(())
        credit = self._credit_in_training_units(batch)
        active = batch["maca_credit"].abs().sum(-1) > 0
        if not active.any():
            return chosen_agent_qvals.new_zeros(())
        lengths = batch["filled"].sum(1).squeeze(-1) - 1
        shaped = terminal_credit_rewards(rewards, credit, lengths)
        transitions = (th.arange(shaped.shape[1], device=shaped.device)[None]
                       < lengths[:, None]).unsqueeze(-1)
        episode_mask = mask * transitions
        targets = finite_episode_returns(shaped, episode_mask, self.args.gamma).detach()
        valid = episode_mask.expand_as(chosen_agent_qvals) * active[:, None, None]
        loss = ((chosen_agent_qvals - targets).square() * valid).sum() / valid.sum()
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("maca/auxiliary_loss", loss.item(), t_env)
            self.logger.log_stat("maca/auxiliary_active_fraction", active.float().mean().item(), t_env)
        return self.auxiliary_weight * loss

    def _checkpoint_config(self):
        names = ("n_agents", "n_actions", "state_shape", "hidden_dim", "agent", "mac",
                 "use_rnn", "obs_last_action", "obs_agent_id", "gamma", "standardise_rewards",
                 "mixing_embed_dim", "hypernet_layers", "hypernet_embed", "double_q",
                 "lr", "grad_norm_clip", "target_update_interval_or_tau", "env")
        return {**{name: getattr(self.args, name) for name in names},
                "env_args": {k: v for k, v in self.args.env_args.items() if k != "seed"},
                "maca": getattr(self.args, "maca", None)}

    def save_models(self, path):
        super().save_models(path)
        th.save({
            "version": 1, "config": self._checkpoint_config(),
            "target_agent": self.target_mac.agent.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "training_steps": self.training_steps,
            "last_target_update_step": self.last_target_update_step,
            "log_stats_t": self.log_stats_t,
            "reward_moments": copy.deepcopy(vars(self.rew_ms)) if self.args.standardise_rewards else None,
            "adapter": self.adapter.state_dict() if self.adapter is not None else None,
        }, Path(path) / "qmix_lbf.th")

    def load_models(self, path):
        device = next(iter(self.mac.parameters())).device
        state = th.load(Path(path) / "qmix_lbf.th", map_location=device, weights_only=False)
        if state["version"] != 1 or state["config"] != self._checkpoint_config():
            raise ValueError("LBF QMIX checkpoint configuration does not match this run")
        super().load_models(path)
        self.target_mac.agent.load_state_dict(state["target_agent"])
        self.target_mixer.load_state_dict(state["target_mixer"])
        self.training_steps = state["training_steps"]
        self.last_target_update_step = state["last_target_update_step"]
        self.log_stats_t = state["log_stats_t"]
        if self.args.standardise_rewards:
            self.rew_ms.__dict__.update(state["reward_moments"])
        if self.adapter is not None:
            self.adapter.load_state_dict(state["adapter"])
