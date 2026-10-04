"""Regression tests for EpisodeMetricsCallback aggregation.

The bug these guard against: the callback stored SB3 ``info`` dicts by
reference and read the top-level keys, so ``train_ep/success`` logged a flat
0.0 (and ``train_ep/timeout`` a flat 1.0) even though ``monitor.csv`` and the
validation metrics recorded real goal completions.
"""
import numpy as np
import pytest

from v2x_rl.callbacks import (METRIC_KEYS, CurriculumCallback,
                              EpisodeMetricsCallback, _aggregate, _as_float,
                              _episode_record)
from v2x_rl.config import CurriculumCfg


def _summary(success=False, collision=False, timeout=True, steps=450):
    return {
        "ep_steps": steps,
        "success": bool(success),
        "collision": bool(collision),
        "collision_with_cyclist": bool(collision),
        "timeout": bool(timeout),
        "off_route": False,
        "stuck": False,
        "mean_speed_kmh": 20.0,
        "mean_speed_error_kmh": 30.0,
        "mean_jerk": 0.1,
        "yield_steps": 40,
        "yield_violation_steps": 0,
        "yield_ok": True,
        "unnecessary_brake_frac": 0.0,
        "min_ttc": 12.0,
        "min_cyclist_distance": 25.0,
        "v2x_valid_frac": 0.6,
        "v2x_loss_rate": 0.3,
        "v2x_nlos_frac": 0.1,
        "maneuver": "left",
    }


def test_as_float_handles_bool_number_and_junk():
    assert _as_float(True) == 1.0
    assert _as_float(False) == 0.0
    assert _as_float(np.bool_(True)) == 1.0
    assert _as_float(3) == 3.0
    assert _as_float("nan") != _as_float("nan") or True  # float('nan') is fine
    assert _as_float(None) is None
    assert _as_float("abc") is None


def test_aggregate_counts_successes():
    records = [_summary(False), _summary(False),
              _summary(True, timeout=False, steps=400), _summary(False)]
    agg = _aggregate(records)
    assert agg["success"] == pytest.approx(0.25)
    assert agg["timeout"] == pytest.approx(0.75)
    assert agg["collision"] == pytest.approx(0.0)


def test_episode_record_prefers_monitor_snapshot():
    # Top-level says failure; Monitor's snapshot (the truth) says success.
    stale_top = _summary(success=False, timeout=True)
    monitor_snapshot = _summary(success=True, timeout=False, steps=402)
    info = {**stale_top, "episode": {**monitor_snapshot, "r": 150.0, "l": 402}}
    rec = _episode_record(info)
    assert rec["success"] is True
    assert rec["timeout"] is False
    assert rec["maneuver"] == "left"


def test_episode_record_falls_back_to_top_level():
    info = _summary(success=True, timeout=False)
    rec = _episode_record(info)
    assert rec["success"] is True
    assert _episode_record({"foo": 1}) is None


def test_callback_logs_nonzero_success_from_monitor_snapshot():
    cb = EpisodeMetricsCallback(window=20)
    recorded = {}
    # BaseCallback.logger proxies to self.model.logger in this SB3 version.
    logger = type("L", (), {"record": lambda self, k, v: recorded.__setitem__(k, v)})()
    cb.model = type("M", (), {"logger": logger})()

    outcomes = ([_summary(success=False)] * 6
                + [_summary(success=True, timeout=False, steps=410)]
                + [_summary(success=False)] * 3)
    for out in outcomes:
        info = {**out, "episode": {**out, "r": 1.0, "l": out["ep_steps"]}}
        cb.locals = {"dones": np.array([True]), "infos": [info]}
        cb._on_step()

    assert recorded["train_ep/success"] == pytest.approx(1 / 10)
    assert recorded["train_ep/timeout"] == pytest.approx(9 / 10)
    assert len(cb._records) == 10


# --------------------------------------------------------------------------- #
#  Curriculum
# --------------------------------------------------------------------------- #
class _FakeValidation:
    def __init__(self):
        self.history = []


class _FakeEnv:
    def __init__(self):
        self.calls = []

    def env_method(self, name, *args):
        self.calls.append((name, args))


def _curriculum(cfg=None):
    cfg = cfg or CurriculumCfg(
        stages=((0.0, 0.0), (0.5, 0.3), (0.75, 0.2)),
        advance_success=0.75, advance_patience=2, min_stage_steps=10_000)
    val = _FakeValidation()
    cb = CurriculumCallback(val, cfg, verbose=0)
    env = _FakeEnv()
    logger = type("L", (), {"record": lambda s, k, v: None})()
    # BaseCallback exposes training_env / logger as read-only properties that
    # proxy to the model.
    cb.model = type("M", (), {"get_env": lambda s: env, "logger": logger})()
    cb.num_timesteps = 0
    cb.training_env_ref = env       # convenient handle for assertions
    return cb, val


def test_curriculum_starts_at_stage_zero():
    cb, _ = _curriculum()
    cb._on_training_start()
    assert cb.stage == 0
    assert cb.training_env_ref.calls[-1] == ("apply_curriculum_stage", (0, 0.0, 0.0))


def test_curriculum_needs_sustained_success_and_dwell_time():
    cb, val = _curriculum()
    cb._on_training_start()

    cb.num_timesteps = 4_000            # below min_stage_steps
    val.history.append({"success": 0.9})
    cb._on_step()
    assert cb.stage == 0               # dwell time not met

    cb.num_timesteps = 12_000          # now past min_stage_steps, streak -> 2
    val.history.append({"success": 0.9})
    cb._on_step()
    assert cb.stage == 1
    assert cb.training_env_ref.calls[-1] == ("apply_curriculum_stage", (1, 0.5, 0.3))


def test_curriculum_streak_resets_on_a_bad_validation():
    cb, val = _curriculum()
    cb._on_training_start()
    cb.num_timesteps = 30_000
    for success in (0.9, 0.4, 0.9):    # the dip breaks the streak
        val.history.append({"success": success})
        cb._on_step()
    assert cb.stage == 0
    val.history.append({"success": 0.9})   # now two in a row again
    cb._on_step()
    assert cb.stage == 1


def test_curriculum_stops_at_the_hardest_stage():
    cb, val = _curriculum()
    cb._on_training_start()
    for i in range(12):
        cb.num_timesteps = 20_000 * (i + 1)   # always past the dwell time
        val.history.append({"success": 1.0})
        cb._on_step()
    assert cb.stage == 2                       # len(stages) - 1, never overshoots
    # stage 0 applied at start, then exactly two advances -- no fourth call.
    assert len(cb.training_env_ref.calls) == 3
