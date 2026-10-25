from types import SimpleNamespace

import numpy as np

from envs.gymma import GymmaWrapper
from runners.parallel_runner import ParallelRunner


def test_parallel_worker_seed_offsets_and_continuing_reset_streams():
    settings = dict(key="lbforaging:Foraging-8x8-2p-2f-coop-v3", time_limit=50,
                    pretrained_wrapper=None, seed=1587)
    args = SimpleNamespace(batch_size_run=3, env="gymma", env_args=settings,
                           common_reward=True, reward_scalarisation="sum")
    def collect():
        runner = ParallelRunner(args, logger=None)
        try:
            states = []
            for _ in range(2):
                for connection in runner.parent_conns:
                    connection.send(("reset", None))
                states.append(np.stack([connection.recv()["state"] for connection in runner.parent_conns]))
            return np.stack(states)
        finally:
            runner.close_env()
            for process in runner.ps:
                process.join(timeout=3)
                if process.is_alive():
                    process.terminate()
                    process.join()
            for connection in (*runner.parent_conns, *runner.worker_conns):
                connection.close()
    first, second = collect(), collect()
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[0, 0], first[0, 1])
    assert not np.array_equal(first[0], first[1])  # Reset continues rather than reseeding.
    for index in range(3):
        env = GymmaWrapper(**{**settings, "seed": 1587 + index}, common_reward=True,
                           reward_scalarisation="sum")
        try:
            for reset in range(2):
                env.reset()
                np.testing.assert_array_equal(first[reset, index], env.get_state())
        finally:
            env.close()
