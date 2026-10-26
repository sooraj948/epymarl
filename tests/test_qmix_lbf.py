import copy
import json
import logging
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.transforms import OneHot
from controllers import REGISTRY as controllers
from learners.q_learner import QLearner
from learners.qmix_lbf_learner import LBFQMIXLearner, finite_episode_returns, terminal_credit_rewards
from utils.logging import Logger


ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def arguments(tmp_path, method="qmix_lbf", beta=.1):
    config = {}
    for name in ("default.yaml", "envs/gymma.yaml", f"algs/{method}.yaml"):
        config.update(yaml.safe_load((ROOT / "src/config" / name).read_text()))
    config.update(n_agents=2, n_actions=6, state_shape=24, device="cpu", use_cuda=False,
                  seed=1587, unique_token=method, local_results_path=str(tmp_path))
    config["env_args"].update(key="lbforaging:Foraging-8x8-2p-2f-coop-v3", time_limit=4, seed=1587)
    if "maca" in config:
        config["maca"].update(beta=beta, mixer_epochs=1, n_samples=2)
    return SimpleNamespace(**config)


def rollout_batch(args, lengths=(2, 4)):
    scheme = {
        "state": {"vshape": 24}, "obs": {"vshape": 12, "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "avail_actions": {"vshape": (6,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)}, "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }
    LBFQMIXLearner.extend_scheme(args, scheme)
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


def learner(args, batch, cls=LBFQMIXLearner, seed=123):
    torch.manual_seed(seed)
    mac = controllers[args.mac](batch.scheme, batch.groups, args)
    return cls(mac, batch.scheme, Logger(logging.getLogger("qmix-test")), args)


def mask_for(batch):
    lengths = batch["filled"].sum(1).squeeze(-1) - 1
    return (torch.arange(batch.max_seq_length - 1)[None] < lengths[:, None]).float().unsqueeze(-1)


def assert_networks_equal(left, right):
    for lnet, rnet in ((left.mac.agent, right.mac.agent), (left.mixer, right.mixer),
                       (left.target_mac.agent, right.target_mac.agent), (left.target_mixer, right.target_mixer)):
        for key, value in lnet.state_dict().items():
            torch.testing.assert_close(value, rnet.state_dict()[key], rtol=0, atol=0)


def test_prescribed_settings_and_matched_architectures(tmp_path):
    stock = yaml.safe_load((ROOT / "src/config/default.yaml").read_text())
    stock.update(yaml.safe_load((ROOT / "src/config/algs/qmix.yaml").read_text()))
    baseline = arguments(tmp_path)
    for key in ("lr", "gamma", "batch_size", "buffer_size", "target_update_interval_or_tau",
                "hidden_dim", "double_q", "mixing_embed_dim", "hypernet_layers", "hypernet_embed",
                "epsilon_start", "epsilon_finish", "epsilon_anneal_time", "evaluation_epsilon",
                "standardise_rewards", "standardise_returns", "grad_norm_clip"):
        assert getattr(baseline, key) == stock[key]
    assert baseline.use_rnn and baseline.obs_last_action and baseline.obs_agent_id
    for method in ("maca_qmix_aux", "maca_qmix_sum"):
        shaped = arguments(tmp_path, method)
        for key, value in vars(baseline).items():
            if key not in {"name", "unique_token"}:
                assert getattr(shaped, key) == value


@pytest.mark.parametrize("method", ["qmix_lbf", "maca_qmix_aux", "maca_qmix_sum"])
def test_baseline_and_beta_zero_match_stock_updates(tmp_path, method):
    args = arguments(tmp_path, method, beta=0.)
    batch = rollout_batch(args)
    model = learner(args, batch)
    stock = learner(args, batch, QLearner)
    assert model.adapter is None
    for t in (1, 2):
        model.prepare_episode(batch, t)
        model.train(batch, t, t)
        stock.train(batch, t, t)
        assert_networks_equal(model, stock)
        torch.testing.assert_close(model.rew_ms.var, stock.rew_ms.var, rtol=0, atol=0)


def test_q_score_unrolls_taken_actions_and_excludes_padding(tmp_path):
    args = arguments(tmp_path, "maca_qmix_aux")
    batch = rollout_batch(args)
    model = learner(args, batch)
    scorer = model.adapter.pipeline.frozen
    saved_hidden = torch.ones(1)
    model.target_mac.hidden_states = saved_hidden
    scores = scorer.score_batch(batch, [2, 4])
    assert model.target_mac.hidden_states is saved_hidden
    with torch.no_grad():
        model.target_mac.init_hidden(2)
        values = []
        for t in range(4):
            q = model.target_mac.forward(batch, t)
            taken = q.gather(2, batch["actions"][:, t]).squeeze(-1)
            values.append(model.target_mixer(taken[:, None], batch["state"][:, t:t+1])[:, 0, 0])
        expected = torch.stack(values, 1)
    np.testing.assert_allclose(scores, [expected[0, :2].mean(), expected[1].mean()], rtol=1e-6)
    altered = copy.deepcopy(batch)
    altered["obs"][0, 2:] = 10000
    altered["state"][0, 2:] = 10000
    np.testing.assert_array_equal(scorer.score_batch(altered, [2, 4]), scores)
    assert all(p.grad is None for p in model.target_mac.parameters())
    assert all(p.grad is None for p in model.target_mixer.parameters())


def test_terminal_placement_and_finite_returns():
    rewards = torch.zeros(2, 4, 1)
    rewards[0, :2, 0] = torch.tensor([1., 2.])
    credit = torch.tensor([[.2, -.1], [.4, -.4]])
    lengths = torch.tensor([2, 4])
    mask = (torch.arange(4)[None] < lengths[:, None]).float().unsqueeze(-1)
    shaped = terminal_credit_rewards(rewards, credit, lengths)
    targets = finite_episode_returns(shaped, mask, .5)
    torch.testing.assert_close(targets[0, 1], torch.tensor([2.2, 1.9]))
    torch.testing.assert_close(targets[0, 0], torch.tensor([2.1, 1.95]))
    assert not targets[0, 2:].any()
    torch.testing.assert_close(targets[1], torch.tensor([[.05, -.05], [.1, -.1], [.2, -.2], [.4, -.4]]))


@pytest.mark.parametrize("method", ["maca_qmix_aux", "maca_qmix_sum"])
def test_collection_once_replay_preserves_raw_rewards_and_rng(tmp_path, monkeypatch, method):
    args = arguments(tmp_path, method)
    batch = rollout_batch(args)
    model = learner(args, batch)
    raw = batch["reward"].clone()
    def signed_credit(*args, **kwargs):
        np.random.rand(3)
        torch.rand(3)
        random.random()
        return np.array([1., -.5]), {"scored": True}
    monkeypatch.setattr(model.adapter.pipeline, "process", signed_credit)
    rng = copy.deepcopy((torch.random.get_rng_state(), np.random.get_state(), random.getstate()))
    model.prepare_episode(batch, 10)
    assert torch.equal(rng[0], torch.random.get_rng_state())
    np.testing.assert_array_equal(rng[1][1], np.random.get_state()[1])
    assert rng[2] == random.getstate()
    assert model.adapter.episodes == 2
    assert batch["maca_credit"].abs().sum() > 0
    replay = ReplayBuffer({k: v for k, v in batch.scheme.items() if k != "filled"},
                          batch.groups, 5, batch.max_seq_length)
    replay.insert_episode_batch(batch)
    sample = replay.sample(2)
    stored = sample["maca_credit"].clone()
    for t in (11, 12):
        model.train(sample, t, t)
    assert model.adapter.episodes == 2
    torch.testing.assert_close(sample["maca_credit"], stored, rtol=0, atol=0)
    torch.testing.assert_close(batch["reward"], raw, rtol=0, atol=0)
    assert model.training_steps == 2


def test_auxiliary_gradients_targets_and_inactive_episodes(tmp_path):
    args = arguments(tmp_path, "maca_qmix_aux")
    args.standardise_rewards = False
    batch = rollout_batch(args)
    batch.update({"maca_credit": torch.tensor([[.2, -.1], [0., 0.]])}, mark_filled=False)
    model = learner(args, batch)
    rewards = batch["reward"][:, :-1].expand(-1, -1, 2)
    chosen = torch.zeros(2, 4, 2, requires_grad=True)
    loss = model._auxiliary_loss(batch, chosen, rewards, mask_for(batch), 1)
    expected = torch.tensor([[[args.gamma * .2, -args.gamma * .1], [.2, -.1]]])
    torch.testing.assert_close(loss, expected.square().mean())
    loss.backward()
    assert chosen.grad[0, :2].abs().sum() > 0
    assert not chosen.grad[0, 2:].any() and not chosen.grad[1].any()
    assert all(p.grad is None for p in model.mixer.parameters())
    prepared = model._prepare_rewards(batch, batch["reward"][:, :-1], mask_for(batch), 1)
    assert not prepared.any()  # Team TD still receives only environment reward.


def test_team_sum_normalization_and_cancellation(tmp_path):
    args = arguments(tmp_path, "maca_qmix_sum")
    args.standardise_rewards = False
    batch = rollout_batch(args)
    model = learner(args, batch)
    batch.update({"maca_credit": torch.tensor([[.2, -.1], [.3, -.3]])}, mark_filled=False)
    raw = batch["reward"][:, :-1]
    prepared = model._prepare_rewards(batch, raw, mask_for(batch), 1)
    torch.testing.assert_close(prepared[0, 1], torch.tensor([.1]))
    assert not prepared[0, 2:].any() and not prepared[1].any()
    assert model._auxiliary_loss(batch, torch.zeros(2, 4, 2), prepared, mask_for(batch), 1) == 0
    args.standardise_rewards = True
    normalized = learner(args, batch)
    stock = learner(args, batch, QLearner)
    expected = stock._prepare_rewards(batch, raw, mask_for(batch), 1)
    actual = normalized._prepare_rewards(batch, raw, mask_for(batch), 1)
    torch.testing.assert_close(actual[0, 1] - expected[0, 1], .1 / stock.rew_ms.var.sqrt())
    torch.testing.assert_close(normalized.rew_ms.var, stock.rew_ms.var, rtol=0, atol=0)


def test_target_score_changes_only_on_target_refresh(tmp_path):
    args = arguments(tmp_path, "maca_qmix_aux")
    args.target_update_interval_or_tau = 2
    batch = rollout_batch(args)
    model = learner(args, batch)
    scorer = model.adapter.pipeline.frozen
    initial = scorer.score_batch(batch, [2, 4])
    model.prepare_episode(batch, 1)
    model.train(batch, 1, 1)
    np.testing.assert_array_equal(scorer.score_batch(batch, [2, 4]), initial)
    model.train(batch, 2, 2)
    assert model.last_target_update_step == 2
    assert not np.array_equal(scorer.score_batch(batch, [2, 4]), initial)
    for live, target in zip(model.mac.parameters(), model.target_mac.parameters()):
        torch.testing.assert_close(live, target, rtol=0, atol=0)
    model.prepare_episode(batch, 3)
    assert all(row["target_version"] == 2 for row in model.adapter.last_ledgers)


def test_auxiliary_timeout_has_no_bootstrap_and_padding_is_excluded(tmp_path):
    args = arguments(tmp_path, "maca_qmix_aux")
    args.standardise_rewards = False
    batch = rollout_batch(args)
    # A timeout is allowed at the prescribed horizon, with no terminal flag.
    batch["terminated"][1, 3] = 0
    batch.update({"maca_credit": torch.tensor([[0., 0.], [.2, -.2]])}, mark_filled=False)
    model = learner(args, batch)
    chosen = torch.zeros(2, 4, 2, requires_grad=True)
    loss = model._auxiliary_loss(batch, chosen, batch["reward"][:, :-1], mask_for(batch), 1)
    expected = torch.tensor([.2 * args.gamma**i for i in (3, 2, 1, 0)])
    torch.testing.assert_close(loss, expected.square().mean())
    model.prepare_episode(batch, 1)
    batch["terminated"][0, 1] = 0  # Missing termination before the configured horizon.
    with pytest.raises(ValueError, match="incomplete"):
        model.prepare_episode(batch, 2)


@pytest.mark.parametrize("method", ["qmix_lbf", "maca_qmix_aux", "maca_qmix_sum"])
def test_checkpoint_restores_targets_normalization_and_next_update(tmp_path, method):
    args = arguments(tmp_path, method)
    batch = rollout_batch(args)
    model = learner(args, batch)
    model.prepare_episode(batch, 1)
    if model.adapter is not None:
        batch.update({"maca_credit": torch.tensor([[.01, -.02], [.03, -.01]])}, mark_filled=False)
    model.train(batch, 1, 1)
    path = tmp_path / "checkpoint"
    path.mkdir()
    model.save_models(path)
    restored = learner(args, batch, seed=456)
    restored.load_models(path)
    assert_networks_equal(model, restored)
    assert restored.training_steps == 1 and restored.last_target_update_step == 0
    for instance in (model, restored):
        instance.train(batch, 2, 2)
    assert_networks_equal(model, restored)
    torch.testing.assert_close(model.rew_ms.var, restored.rew_ms.var, rtol=0, atol=0)
    if model.adapter is not None:
        assert restored.adapter.episodes == model.adapter.episodes == 2
        manifest = json.loads((model.adapter.output / "manifest.json").read_text())
        assert manifest["teacher_score"] == "mean_target_q_tot_recorded_actions"
        assert manifest["replay_credit"] == "fixed_at_collection_never_rescored"


def test_checkpoint_rejects_different_shaping_mode(tmp_path):
    args = arguments(tmp_path, "maca_qmix_aux")
    batch = rollout_batch(args)
    model = learner(args, batch)
    path = tmp_path / "checkpoint"
    path.mkdir()
    model.save_models(path)
    other = arguments(tmp_path, "maca_qmix_sum")
    restored = learner(other, rollout_batch(other))
    with pytest.raises(ValueError, match="configuration"):
        restored.load_models(path)


def test_suite_matrix(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_qmix_suite import parse_args, stages_for
    stages = stages_for(parse_args([]), tmp_path)
    assert [len(s["jobs"]) for s in stages] == [12, 12, 12]
    assert sum(j.budget_steps for s in stages for j in s["jobs"]) == 96_000_000
    stages = stages_for(parse_args(["--smoke", "--tasks", "8x8_2p2f_coop"]), tmp_path)
    assert len(stages) == 1 and len(stages[0]["jobs"]) == 3
    assert {j.seed for j in stages[0]["jobs"]} == {1587}
    assert all(j.budget_steps == 6000 for j in stages[0]["jobs"])
