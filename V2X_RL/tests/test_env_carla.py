"""Integration tests that need a running CARLA server.

Run with::

    pytest tests/test_env_carla.py -m carla -v

They are skipped automatically when no simulator is listening.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.carla


def _carla_available() -> bool:
    try:
        from v2x_rl.carla_utils import carla
        client = carla.Client("localhost", 2000)
        client.set_timeout(4.0)
        client.get_server_version()
        return True
    except Exception:
        return False


pytest.importorskip("carla")
if not _carla_available():
    pytest.skip("no CARLA server on localhost:2000", allow_module_level=True)


from v2x_rl.config import EnvCfg  # noqa: E402
from v2x_rl.envs import IntersectionV2XEnv  # noqa: E402


@pytest.fixture(scope="module")
def cfg() -> EnvCfg:
    # A short, lidar-only episode keeps the test fast.
    return EnvCfg().merge({
        "scenario.max_episode_steps": 60,
        "sensors.depth.enabled": False,
        "sensors.lidar.enabled": True,
        "carla.no_rendering": True,
    })


@pytest.fixture(scope="module")
def env(cfg):
    environment = IntersectionV2XEnv(cfg)
    yield environment
    environment.close()


def test_passes_gymnasium_api_checker(cfg):
    from stable_baselines3.common.env_checker import check_env
    environment = IntersectionV2XEnv(cfg)
    try:
        check_env(environment, warn=True, skip_render_check=True)
    finally:
        environment.close()


def test_reset_returns_observation_within_the_declared_space(env):
    observation, info = env.reset(seed=1)
    assert env.observation_space.contains(observation)
    assert "maneuver" in info


def test_step_advances_and_stays_in_bounds(env):
    env.reset(seed=2)
    for _ in range(20):
        observation, reward, terminated, truncated, info = env.step(
            np.array([0.5], dtype=np.float32))
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        if terminated or truncated:
            break


def test_episode_terminates_and_reports_metrics(env):
    env.reset(seed=3)
    info = {}
    for _ in range(env.cfg.scenario.max_episode_steps + 5):
        _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        if terminated or truncated:
            break
    assert terminated or truncated
    for key in ("ep_steps", "success", "collision", "mean_speed_kmh",
                "v2x_loss_rate", "min_ttc"):
        assert key in info, f"missing metric {key}"


def test_same_seed_reproduces_the_same_layout(env):
    first, _ = env.reset(seed=7)
    maneuver_a = env.layout.maneuver
    cyclist_a = env.layout.cyclist_present
    second, _ = env.reset(seed=7)
    assert env.layout.maneuver == maneuver_a
    assert env.layout.cyclist_present == cyclist_a


def test_all_three_maneuvers_are_reachable(env):
    seen = set()
    for seed in range(25):
        env.reset(seed=seed)
        seen.add(env.layout.maneuver)
        if len(seen) == 3:
            break
    assert seen == {"left", "right", "straight"}, f"only saw {seen}"


def test_conflicts_and_clear_runs_both_occur(env):
    outcomes = set()
    for seed in range(30):
        env.reset(seed=seed)
        outcomes.add(bool(env.layout.has_conflict))
        if len(outcomes) == 2:
            break
    assert outcomes == {True, False}, (
        "expected both conflicting and non-conflicting episodes, got "
        f"{outcomes}")


def test_action_rate_limit_smooths_the_command(cfg):
    limited = IntersectionV2XEnv(cfg.merge({"action_rate_limit": 0.1}))
    try:
        limited.reset(seed=11)
        limited.step(np.array([1.0], dtype=np.float32))
        # The applied command cannot jump from 0 to 1 in a single step.
        assert limited._prev_action == pytest.approx(0.1)
    finally:
        limited.close()
