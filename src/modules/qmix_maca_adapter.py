"""Collection-time MACA credit with recurrent, taken-action team-Q labels."""

import json

import numpy as np
import torch

from maca.trajectory_logging import to_jsonable
from modules.maca_adapter import MACAAdapter, state_from_observations


class QMIXScorer:
    def __init__(self, networks, config):
        self.mac, self.mixer = networks
        self.config = config
        # Common adapter checkpoint interface; Q labels need no reward transform.
        self.set_reward_normalization(0., 1.)

    def set_reward_normalization(self, mean, std):
        self.reward_mean, self.reward_std = float(mean), float(std)

    def score(self, states):
        raise ValueError("QMIX requires a complete episode with recorded actions")

    @torch.no_grad()
    def score_batch(self, batch, lengths):
        """Reset and unroll target history, excluding final observations/padding."""
        hidden = self.mac.hidden_states
        try:
            self.mac.init_hidden(batch.batch_size)
            outputs = torch.stack([
                self.mac.forward(batch, t) for t in range(batch.max_seq_length - 1)
            ], dim=1)
            taken = outputs.gather(3, batch["actions"][:, :-1]).squeeze(3)
            joint = self.mixer(taken, batch["state"][:, :-1]).squeeze(-1)
            lengths = torch.as_tensor(lengths, device=joint.device)
            valid = torch.arange(joint.shape[1], device=joint.device)[None] < lengths[:, None]
            scores = joint.masked_fill(~valid, 0.).sum(1) / lengths
            if not torch.isfinite(scores).all():
                raise ValueError("QMIX trajectory scores must be finite")
            return scores.cpu().numpy()
        finally:
            self.mac.hidden_states = hidden


class QMIXMACAAdapter(MACAAdapter):
    options = (MACAAdapter.options - {"ppo_gae_lambda", "reward_standardization"}) | {
        "mode", "auxiliary_weight",
    }

    def _validate_algorithm(self, args):
        if (args.mixer != "qmix" or args.mac != "basic_mac" or args.agent != "rnn"
                or not args.use_rnn or args.agent_output_type != "q"):
            raise ValueError("QMIX MACA requires shared recurrent QMIX utilities")
        if args.maca.get("mode") not in {"individual_aux", "team_sum"}:
            raise ValueError("QMIX MACA mode must be individual_aux or team_sum")
        weight = args.maca.get("auxiliary_weight", 1.)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("auxiliary_weight must be finite and nonnegative")

    def _configure_algorithm(self, c, args):
        c.online_critic_score = c.mean_advantage_score = c.gae_advantage_score = False
        c.negate_mean_advantage_score = False
        c.occupancy_goal_input = True
        c.r_int_injection = "last_step"

    def _make_scorer(self, networks, config):
        return QMIXScorer(networks, config)

    def _manifest_details(self, args):
        return {
            "variant": "qmix_online_random_donor_splice",
            "mode": args.maca["mode"],
            "teacher_score": "mean_target_q_tot_recorded_actions",
            "teacher_update": "existing_qmix_target_updates",
            "teacher_units": "qmix_environment_reward_normalization_units",
            "teacher_independent_of_shaping": False,
            "counterfactual_score": "occupancy_goal_regressor_no_simulator_replay",
            "replay_credit": "fixed_at_collection_never_rescored",
            "regression_labels": "fixed_collection_scores_across_target_versions",
            "policy_shaping": "terminal_bonus",
            "reward_moments_source": "environment_only_stock_qmix_replay_updates",
            "auxiliary_target": "finite_episode_discounted_shaped_return_no_bootstrap",
            "auxiliary_active": "episodes_with_nonzero_applied_individual_credit",
            "team_target": "environment_only" if args.maca["mode"] == "individual_aux" else "environment_plus_summed_terminal_credit",
        }

    def prepare(self, batch, t_env, target_version):
        """Called once before insertion into replay, never from learner.train()."""
        c = self.config
        lengths = []
        episodes = []
        # Validate all episodes before advancing any persistent shaping state.
        for b in range(batch.batch_size):
            filled = batch["filled"][b, :, 0].detach().cpu().numpy()
            count = int(filled.sum())
            if count < 2 or not np.array_equal(filled, np.r_[np.ones(count), np.zeros(len(filled) - count)]):
                raise ValueError("QMIX MACA requires complete contiguous episodes and a final observation")
            length = count - 1
            if length > c.max_ep_len or batch["terminated"][b, :length - 1].any():
                raise ValueError("Invalid QMIX MACA episode boundary")
            if length < c.max_ep_len and not batch["terminated"][b, length - 1].any():
                raise ValueError("QMIX MACA cannot score an incomplete episode")
            obs = batch["obs"][b, :count].detach().cpu().numpy()
            if not np.array_equal(batch["state"][b, :count].detach().cpu().numpy(), obs.reshape(count, -1)):
                raise ValueError("QMIX MACA needs concatenated native observations as state")
            actions = batch["actions"][b, :length, :, 0]
            if (actions < 0).any() or (actions >= c.action_dim).any():
                raise ValueError("Invalid requested LBF action")
            available = batch["avail_actions"][b, :length].gather(-1, actions.unsqueeze(-1))
            if not available.all():
                raise ValueError("Recorded QMIX action is unavailable")
            actions = actions.detach().cpu().numpy()
            rewards = batch["reward"][b, :length, 0].detach().cpu().numpy()
            if not np.isfinite(rewards).all():
                raise ValueError("Nonfinite environment reward")
            states = [state_from_observations(o, c) for o in obs]
            trajectories = [[(obs[t, i].copy(), int(actions[t, i])) for t in range(length)]
                            for i in range(c.n_agents)]
            episodes.append((trajectories, states, rewards))
            lengths.append(length)
        scores = self.pipeline.frozen.score_batch(batch, lengths)
        applied = batch["reward"].new_zeros(batch.batch_size, c.n_agents)
        ledgers = []
        with self.isolated_rng():
            for b, (trajectories, states, rewards) in enumerate(episodes):
                env_return = float(rewards.sum())
                self.returns.append(env_return)
                average = float(np.mean(self.returns))
                credit, ledger = self.pipeline.process(
                    self.episodes, trajectories, states[:-1], states[-1], env_return,
                    average, None, None, None, None, team_rewards=rewards,
                    episode_score=scores[b],
                )
                fraction = (float(average > c.beta_threshold) if c.beta_ramp == 0 else
                            float(np.clip((average - c.beta_threshold) / c.beta_ramp, 0, 1)))
                weight = c.beta * fraction
                applied[b] = torch.as_tensor(credit, device=applied.device) * weight
                ledger.update(
                    t_env=t_env, target_version=target_version, episode_length=lengths[b],
                    env_return=env_return, running_return=average, effective_weight=weight,
                    score_type="mean_target_q_tot_recorded_actions", injection="last_step",
                    applied_credit=applied[b].cpu().tolist(),
                    summed_credit=float(applied[b].sum()),
                    absolute_credit_sum=float(applied[b].abs().sum()),
                )
                ledgers.append(ledger)
                self.episodes += 1
        if not torch.isfinite(applied).all():
            raise ValueError("QMIX MACA produced nonfinite credit")
        self.last_ledgers = ledgers
        with (self.output / "credit_episodes.jsonl").open("a") as stream:
            stream.writelines(json.dumps(to_jsonable(row)) + "\n" for row in ledgers)
        for name, value in {
            "episodes": self.episodes, "fit_version": self.pipeline.fit_version,
            "scored_fraction": np.mean([r["scored"] for r in ledgers]),
            "effective_beta": np.mean([r["effective_weight"] for r in ledgers]),
            "summed_credit": float(applied.sum(-1).mean()),
            "absolute_credit_sum": float(applied.abs().sum(-1).mean()),
            "collection_q_score": float(np.mean(scores)),
        }.items():
            self.logger.log_stat("maca/" + name, float(value), t_env)
        return applied
