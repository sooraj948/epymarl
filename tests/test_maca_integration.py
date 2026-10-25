import copy
import json
import logging
from pathlib import Path
import random
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers import REGISTRY as controllers
from learners.maca_ppo_learner import MACAPPOLearner
from learners.ppo_learner import PPOLearner
from maca.credit_pipelines import observations_from_state
from maca.config import MACAConfig
from modules.maca_adapter import MACAAdapter, state_from_observations
from utils.logging import Logger


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def arguments(tmp_path, beta=0.1, token="test", shared=False):
    config = {}
    for name in ("default.yaml", "envs/gymma.yaml", "algs/maca_mappo_ns.yaml"):
        config.update(yaml.safe_load((ROOT / "src/config" / name).read_text()))
    config.update(n_agents=2, n_actions=6, state_shape=24, device="cpu", seed=1587,
                  batch_size_run=2, batch_size=2, buffer_size=2, unique_token=token,
                  local_results_path=str(tmp_path))
    config["env_args"]["time_limit"] = 4
    config["env_args"]["seed"] = config["seed"]
    config["maca"].update(beta=beta, fit_freq=1, mixer_epochs=1, n_samples=2)
    if shared:
        config.update(yaml.safe_load((ROOT / "src/config/algs/mappo.yaml").read_text()))
        config.update(mac="basic_mac", agent="rnn")
    return SimpleNamespace(**config)


def rollout_batch(args, lengths=(2, 4)):
    scheme = {
        "state": {"vshape": 24},
        "obs": {"vshape": 12, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (6,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    batch = EpisodeBatch(scheme, {"agents": 2}, len(lengths), max(lengths) + 1,
                         preprocess={"actions": ("actions_onehot", [OneHot(6)])})
    env = gym.make(args.env_args["key"])
    try:
        for b, length in enumerate(lengths):
            obs, _ = env.reset(seed=args.seed + b)
            for t in range(length + 1):
                batch.update({"obs": [obs], "state": [np.concatenate(obs)],
                              "avail_actions": [np.ones((2, 6), dtype=int)]}, bs=b, ts=t)
                if t == length:
                    break
                actions = [0, 0]
                obs, rewards, _, _, _ = env.step(actions)
                batch.update({"actions": torch.tensor(actions).reshape(1, 2, 1),
                              "reward": [(float(sum(rewards)),)],
                              "terminated": [(int(t == length - 1),)]}, bs=b, ts=t)
    finally:
        env.close()
    return batch


def learner(args, batch, cls=MACAPPOLearner):
    mac = controllers[args.mac](batch.scheme, batch.groups, args)
    return cls(mac, batch.scheme, Logger(logging.getLogger("test")), args)


def rng_snapshot():
    return copy.deepcopy((np.random.get_state(), torch.random.get_rng_state(), random.getstate()))


def assert_rng_equal(left, right):
    assert left[0][0] == right[0][0]
    np.testing.assert_array_equal(left[0][1], right[0][1])
    assert left[0][2:] == right[0][2:]
    assert torch.equal(left[1], right[1])
    assert left[2] == right[2]


def test_native_observation_roundtrip_and_teacher(tmp_path):
    args = arguments(tmp_path)
    batch = rollout_batch(args)
    model = learner(args, batch)
    adapter = model.adapter
    observations = batch["obs"][1, :5].numpy()
    states = [state_from_observations(obs, adapter.config) for obs in observations]
    for obs, state in zip(observations, states):
        np.testing.assert_array_equal(observations_from_state(state, adapter.config), obs)
    with torch.no_grad():
        direct = model.critic(batch)[1, :5, :, 0].mean(-1).numpy()
    np.testing.assert_allclose(adapter.pipeline.frozen.values(states), direct, atol=1e-7)
    rewards = np.array([0., 0., 0., 1.])
    gae, advantages = 0., []
    for t in reversed(range(4)):
        bootstrap = direct[t + 1] if t < 3 else 0.
        gae = rewards[t] + args.gamma * bootstrap - direct[t] + args.gamma * .95 * gae
        advantages.append(gae)
    expected = -np.mean(advantages)
    assert adapter.pipeline.frozen.advantage_score(states[:4], rewards) == pytest.approx(expected)
    assert all(p.grad is None for p in model.critic.parameters())
    bad = observations[0].copy()
    bad[1, :3] = [-1, -1, 0]
    with pytest.raises(ValueError, match="ordering"):
        state_from_observations(bad, adapter.config)


def test_signed_uniform_rewards_padding_and_rng_isolation(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    batch = rollout_batch(args)
    model = learner(args, batch, PPOLearner)
    policy_rng = rng_snapshot()
    adapter = MACAAdapter(args, batch.scheme, model.critic, model.logger)
    assert_rng_equal(policy_rng, rng_snapshot())
    def signed_credit(*positional, **kwargs):
        np.random.rand(5)
        torch.rand(5)
        random.random()
        return np.array([1., -1.]), {"scored": True}
    monkeypatch.setattr(adapter.pipeline, "process", signed_credit)
    original = batch["reward"].clone()
    shaped = adapter.prepare(batch, 1)
    assert_rng_equal(policy_rng, rng_snapshot())
    assert torch.equal(original, batch["reward"])
    for b, length in enumerate((2, 4)):
        beta = adapter.last_ledgers[b]["effective_weight"]
        np.testing.assert_allclose(shaped[b, :length].sum(0).numpy(), [beta, -beta], atol=1e-8)
        assert not shaped[b, length:].any()
    assert adapter.episodes == 2


@pytest.mark.parametrize("shared", [False, True])
def test_beta_zero_matches_baseline_parameters_and_updates(tmp_path, shared):
    args = arguments(tmp_path, beta=0, shared=shared)
    batch = rollout_batch(args)
    torch.manual_seed(10)
    baseline = learner(args, batch, PPOLearner)
    torch.manual_seed(10)
    maca = learner(args, batch)
    assert maca.adapter is None
    baseline.train(batch, 8, 0)
    maca.train(batch, 8, 0)
    for left, right in zip(baseline.mac.parameters(), maca.mac.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(baseline.critic.parameters(), maca.critic.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    if shared:
        torch.testing.assert_close(baseline.rew_ms.mean, maca.rew_ms.mean, rtol=0, atol=0)
        torch.testing.assert_close(baseline.rew_ms.var, maca.rew_ms.var, rtol=0, atol=0)
        assert baseline.rew_ms.count == maca.rew_ms.count


@pytest.mark.parametrize("shared", [False, True])
def test_real_credit_update_and_checkpoint_roundtrip(tmp_path, monkeypatch, shared):
    args = arguments(tmp_path, shared=shared)
    batch = rollout_batch(args)
    torch.manual_seed(21)
    model = learner(args, batch)
    # A repeated short fixture supplies the real pipeline's 50-episode readiness quota.
    for step in range(26):
        model.adapter.prepare(batch, step)
    assert model.adapter.pipeline.fit_version > 0
    assert any(row["scored"] for row in model.adapter.last_ledgers)
    assert any(abs(x) > 0 for row in model.adapter.last_ledgers for x in row["applied_credit"])
    counted = []
    original = model.adapter.pipeline.process
    def process(*positional, **kwargs):
        counted.append(1)
        return original(*positional, **kwargs)
    monkeypatch.setattr(model.adapter.pipeline, "process", process)
    before = [p.detach().clone() for p in model.mac.parameters()]
    model.train(batch, 108, 52)
    assert len(counted) == batch.batch_size  # Independent of PPO epoch count.
    assert any(not torch.equal(a, b) for a, b in zip(before, model.mac.parameters()))
    assert all(torch.isfinite(p).all() for p in model.mac.parameters())
    assert model.shaped_rewards is None
    monkeypatch.undo()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model.save_models(checkpoint)
    restored = learner(arguments(tmp_path, token="restored", shared=shared), batch)
    restored.load_models(checkpoint)
    assert restored.adapter.episodes == model.adapter.episodes
    assert restored.critic_training_steps == model.critic_training_steps
    if shared:
        torch.testing.assert_close(restored.rew_ms.mean, model.rew_ms.mean, rtol=0, atol=0)
        torch.testing.assert_close(restored.rew_ms.var, model.rew_ms.var, rtol=0, atol=0)
        assert restored.rew_ms.count == model.rew_ms.count
    expected = model.adapter.prepare(batch, 116)
    actual = restored.adapter.prepare(batch, 116)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.adapter.last_ledgers == model.adapter.last_ledgers
    assert all(a["critic_source"] == "online" for a in model.adapter.last_ledgers)
    rows = [json.loads(line) for line in (model.adapter.output / "credit_episodes.jsonl").read_text().splitlines()]
    assert len(rows) == model.adapter.episodes
    assert (model.adapter.output / "manifest.json").exists()


@pytest.mark.parametrize("change,message", [
    ({"standardise_returns": True}, "standardise_returns=False"),
    ({"buffer_size": 10}, "fresh episode"),
    ({"obs_last_action": True}, "state-only"),
    ({"common_reward": False}, "common_reward"),
])
def test_reject_unsupported_comparisons(tmp_path, change, message):
    args = arguments(tmp_path)
    batch = rollout_batch(args)
    for name, value in change.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match=message):
        learner(args, batch)


def test_comparison_yaml_settings_match():
    path = ROOT / "src/config/algs"
    baseline = yaml.safe_load((path / "maca_baseline_ns.yaml").read_text())
    maca = yaml.safe_load((path / "maca_mappo_ns.yaml").read_text())
    for key in ("name", "learner", "maca"):
        baseline.pop(key, None)
        maca.pop(key, None)
    assert baseline == maca


def test_shared_yaml_preserves_stock_mappo_settings():
    path = ROOT / "src/config/algs"
    baseline = yaml.safe_load((path / "mappo.yaml").read_text())
    maca = yaml.safe_load((path / "maca_mappo.yaml").read_text())
    for key in ("name", "learner", "maca"):
        baseline.pop(key, None)
        maca.pop(key, None)
    assert baseline == maca


def test_shared_shaping_normalization_and_teacher_units(tmp_path, monkeypatch):
    args = arguments(tmp_path, shared=True)
    batch = rollout_batch(args)
    batch["reward"][1, 3, 0] = .5
    original = batch["reward"].clone()
    model = learner(args, batch)
    before_values = model.critic(batch).detach()[:, :, :, 0].mean(-1).numpy()
    scores, captured = [], []
    def signed_credit(ep, trajectories, states, final_state, env_return, running_avg,
                      env, snapshot, initial_obs, agents, team_rewards):
        scores.append(model.adapter.pipeline.frozen.advantage_score(states, team_rewards))
        return np.array([1., -1.]), {"scored": True}
    monkeypatch.setattr(model.adapter.pipeline, "process", signed_credit)
    original_train = PPOLearner.train_critic_sequential
    def record_rewards(self, critic, target_critic, batch, rewards, mask):
        captured.append(rewards.detach().clone())
        return original_train(self, critic, target_critic, batch, rewards, mask)
    monkeypatch.setattr(PPOLearner, "train_critic_sequential", record_rewards)
    policy_rng = rng_snapshot()
    model.train(batch, 8, 0)
    assert_rng_equal(policy_rng, rng_snapshot())
    assert torch.equal(batch["reward"], original)
    assert model.rew_ms.count == pytest.approx(1e-4 + 8)  # One update, including upstream padding.
    mean, std = model.rew_ms.mean.item(), model.rew_ms.var.sqrt().item()
    assert model.adapter.pipeline.frozen.reward_mean == pytest.approx(mean)
    assert model.adapter.pipeline.frozen.reward_std == pytest.approx(std)
    raw_shaped = original[:, :-1].expand(-1, -1, 2).clone()
    for b, length in enumerate((2, 4)):
        weight = model.adapter.last_ledgers[b]["effective_weight"]
        raw_shaped[b, :length] += torch.tensor([weight, -weight]) / length
        rewards = (original[b, :length, 0].numpy() - mean) / std
        gae, advantages = 0., []
        for t in reversed(range(length)):
            next_value = before_values[b, t + 1] if t + 1 < length else 0.
            gae = rewards[t] + args.gamma * next_value - before_values[b, t] + args.gamma * .95 * gae
            advantages.append(gae)
        assert scores[b] == pytest.approx(-np.mean(advantages), abs=1e-6)
    assert len(captured) == args.epochs
    for rewards in captured:
        torch.testing.assert_close(rewards, (raw_shaped - mean) / std)
    assert not torch.equal(captured[0][:, :, 0], captured[0][:, :, 1])


def test_zero_credit_shared_update_matches_stock_mappo(tmp_path, monkeypatch):
    args = arguments(tmp_path, shared=True)
    batch = rollout_batch(args)
    batch["reward"][1, 3, 0] = .5
    torch.manual_seed(17)
    baseline = learner(args, batch, PPOLearner)
    torch.manual_seed(17)
    maca = learner(args, batch)
    monkeypatch.setattr(maca.adapter.pipeline, "process",
                        lambda *a, **kw: (np.zeros(2), {"scored": False}))
    baseline.train(batch, 8, 0)
    maca.train(batch, 8, 0)
    for left, right in zip(baseline.mac.parameters(), maca.mac.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    for left, right in zip(baseline.critic.parameters(), maca.critic.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_native_three_player_blocked_chain_observation_roundtrip():
    # Native LBF resolves collisions at destinations independently: the first
    # player can move onto the second when the second is blocked by the third.
    key = "lbforaging:Foraging-8x8-3p-2f-v3"
    env = gym.make(key)
    try:
        env.reset(seed=1587)
        base = env.unwrapped
        base.field[:] = 0
        base.field[5, 5] = base.field[6, 6] = 1
        for player, position in zip(base.players, [(1, 1), (1, 2), (1, 3)]):
            player.position = position
        base._gen_valid_moves()
        observations, *_ = env.step([4, 4, 0])
        assert base.players[0].position == base.players[1].position
        config = MACAConfig.from_env_name(key)
        state = state_from_observations(observations, config)
        np.testing.assert_array_equal(observations_from_state(state, config), observations)
    finally:
        env.close()
