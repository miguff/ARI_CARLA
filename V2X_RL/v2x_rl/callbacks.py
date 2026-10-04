"""Training callbacks: periodic validation and episode metric logging."""
from __future__ import annotations

import csv
import logging
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

LOGGER = logging.getLogger(__name__)

# Episode diagnostics reported by the environment in the terminal ``info``.
METRIC_KEYS = [
    "ep_steps",
    "success",
    "collision",
    "collision_with_cyclist",
    "timeout",
    "off_route",
    "stuck",
    "mean_speed_kmh",
    "mean_speed_error_kmh",
    "mean_jerk",
    "mean_residual",
    "yield_steps",
    "yield_violation_steps",
    "yield_ok",
    "unnecessary_brake_frac",
    "min_ttc",
    "min_cyclist_distance",
    "min_ttc_conflict",
    "min_cyclist_distance_conflict",
    "curriculum_stage",
    "v2x_valid_frac",
    "v2x_loss_rate",
    "v2x_nlos_frac",
]


def _as_float(value: Any) -> Optional[float]:
    """Coerce a metric value (number, bool, numpy scalar) to float, or None."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Mean of every numeric/boolean metric across episodes."""
    if not records:
        return {}
    out: Dict[str, float] = {}
    for key in METRIC_KEYS:
        values = [f for f in (_as_float(r[key]) for r in records if key in r)
                  if f is not None]
        if values:
            out[key] = float(np.mean(values))
    return out


def _episode_record(info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract the per-episode metric dict from a step ``info``.

    ``stable_baselines3``'s ``Monitor`` copies the environment's terminal
    metrics into ``info['episode']`` at the exact moment the episode ends (the
    same values it writes to ``monitor.csv``).  Those are authoritative;
    reading the top-level keys instead has been observed to pick up stale
    ``success`` / ``timeout`` values by the time this callback aggregates.
    Fall back to the top-level keys only if the Monitor snapshot is absent.
    """
    episode = info.get("episode")
    if isinstance(episode, dict) and "ep_steps" in episode:
        source = episode
    elif "ep_steps" in info:
        source = info
    else:
        return None
    keys = METRIC_KEYS + ["maneuver"]
    return {k: source[k] for k in keys if k in source}


class EpisodeMetricsCallback(BaseCallback):
    """Logs a rolling mean of the environment's episode diagnostics."""

    def __init__(self, window: int = 20, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = window
        self._records: deque = deque(maxlen=window)
        self._per_maneuver: Dict[str, deque] = {}

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []),
                              self.locals.get("infos", [])):
            if not done:
                continue
            record = _episode_record(info)
            if record is None:
                continue
            # Store a detached copy: the SB3 info dicts must not be retained by
            # reference across steps or fields can go stale before aggregation.
            self._records.append(record)
            maneuver = record.get("maneuver", "unknown")
            self._per_maneuver.setdefault(maneuver, deque(maxlen=self.window))
            self._per_maneuver[maneuver].append(record)

        if self._records:
            for key, value in _aggregate(self._records).items():
                self.logger.record(f"train_ep/{key}", value)
            for maneuver, records in self._per_maneuver.items():
                summary = _aggregate(records)
                for key in ("success", "collision", "yield_ok", "mean_speed_kmh"):
                    if key in summary:
                        self.logger.record(f"train_{maneuver}/{key}", summary[key])
        return True


class PeriodicValidationCallback(BaseCallback):
    """Runs a deterministic validation phase and saves *every* checkpoint.

    ``EvalCallback`` from stable-baselines3 only keeps the single best model;
    here every validation snapshot is written to disk (model + observation
    normalisation statistics) together with a CSV row of its metrics, so any
    intermediate policy can be re-evaluated later.

    Validation runs on the training environment.  A CARLA server hosts one
    world, and spawning a second ego vehicle in it would let the two
    environments interfere, so a separate evaluation env is not an option.
    The consequence is that the training episode in progress is abandoned at
    each validation; the agent's observation is re-synchronised afterwards.
    """

    def __init__(self, venv, save_dir: str, n_episodes: int = 8,
                 eval_freq: int = 10_000, base_seed: int = 10_000,
                 deterministic: bool = True, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.venv = venv
        self.save_dir = save_dir
        self.n_episodes = n_episodes
        self.eval_freq = eval_freq
        self.base_seed = base_seed
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.history: List[Dict[str, Any]] = []
        self._last_validated_step = -1
        os.makedirs(save_dir, exist_ok=True)
        self.csv_path = os.path.join(save_dir, "validation_log.csv")

    # ------------------------------------------------------------------ #
    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True
        self.validate()
        return True

    def _on_training_end(self) -> None:
        """Score the final policy, unless it was just validated anyway.

        When ``total_timesteps`` is a multiple of ``eval_freq`` the periodic
        validation has already run at this exact step, and repeating it would
        only add a duplicate row and a redundant checkpoint.
        """
        if self._last_validated_step == self.num_timesteps:
            return
        self.validate(tag="final")

    # ------------------------------------------------------------------ #
    def validate(self, tag: Optional[str] = None) -> Dict[str, Any]:
        self._last_validated_step = self.num_timesteps
        vecnorm = self._find_vecnormalize()
        saved_flags = None
        if vecnorm is not None:
            saved_flags = (vecnorm.training, vecnorm.norm_reward)
            vecnorm.training = False       # freeze the running statistics
            vecnorm.norm_reward = False    # report raw, comparable returns

        started = time.time()
        returns: List[float] = []
        lengths: List[int] = []
        records: List[Dict[str, Any]] = []

        try:
            for episode in range(self.n_episodes):
                self.venv.seed(self.base_seed + episode)
                obs = self.venv.reset()
                total, steps, done = 0.0, 0, False
                state = None
                while not done:
                    action, state = self.model.predict(
                        obs, state=state, deterministic=self.deterministic)
                    obs, reward, dones, infos = self.venv.step(action)
                    total += float(reward[0])
                    steps += 1
                    done = bool(dones[0])
                    if done and "ep_steps" in infos[0]:
                        records.append(infos[0])
                returns.append(total)
                lengths.append(steps)
        finally:
            if vecnorm is not None and saved_flags is not None:
                vecnorm.training, vecnorm.norm_reward = saved_flags
            self._resync_training_env()

        summary: Dict[str, Any] = {
            "timesteps": int(self.num_timesteps),
            "mean_reward": float(np.mean(returns)) if returns else 0.0,
            "std_reward": float(np.std(returns)) if returns else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "n_episodes": len(returns),
            "wall_time_s": time.time() - started,
        }
        summary.update(_aggregate(records))

        for key, value in summary.items():
            if key != "timesteps":
                self.logger.record(f"validation/{key}", value)
        self.logger.dump(self.num_timesteps)

        path = self._save(summary, tag)
        summary["path"] = path
        self.history.append(summary)
        self._append_csv(summary)

        if self.verbose:
            LOGGER.info(
                "[validation @ %d] reward %.2f +/- %.2f | success %.0f%% | "
                "collision %.0f%% | yield_ok %.0f%% | speed %.1f km/h | -> %s",
                summary["timesteps"], summary["mean_reward"], summary["std_reward"],
                100 * summary.get("success", 0.0), 100 * summary.get("collision", 0.0),
                100 * summary.get("yield_ok", 0.0), summary.get("mean_speed_kmh", 0.0),
                os.path.basename(path))
        return summary

    # ------------------------------------------------------------------ #
    def _save(self, summary: Dict[str, Any], tag: Optional[str]) -> str:
        stem = (f"val_step{summary['timesteps']:08d}"
                f"_r{summary['mean_reward']:+.1f}"
                f"_col{summary.get('collision', 0.0):.2f}"
                f"_suc{summary.get('success', 0.0):.2f}")
        if tag:
            stem = f"{stem}_{tag}"
        model_path = os.path.join(self.save_dir, f"{stem}.zip")
        self.model.save(model_path)

        vecnorm = self._find_vecnormalize()
        if vecnorm is not None:
            vecnorm.save(os.path.join(self.save_dir, f"{stem}_vecnormalize.pkl"))

        if summary["mean_reward"] > self.best_mean_reward:
            self.best_mean_reward = summary["mean_reward"]
            self.model.save(os.path.join(self.save_dir, "best_model.zip"))
            if vecnorm is not None:
                vecnorm.save(os.path.join(self.save_dir, "best_vecnormalize.pkl"))
        return model_path

    def _append_csv(self, summary: Dict[str, Any]) -> None:
        fields = ["timesteps", "mean_reward", "std_reward", "mean_length",
                  "n_episodes", "wall_time_s", *METRIC_KEYS, "path"]
        row = {key: summary.get(key, "") for key in fields}
        write_header = not os.path.isfile(self.csv_path)
        with open(self.csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _find_vecnormalize(self) -> Optional[VecNormalize]:
        env = self.venv
        while env is not None:
            if isinstance(env, VecNormalize):
                return env
            env = getattr(env, "venv", None)
        return None

    def _resync_training_env(self) -> None:
        """Point the learner at a fresh episode after validation.

        The validation loop left the environment mid-episode (or just reset),
        so the algorithm's cached observation no longer matches reality.
        """
        obs = self.venv.reset()
        self.model._last_obs = obs
        n_envs = self.venv.num_envs
        self.model._last_episode_starts = np.ones((n_envs,), dtype=bool)
        if hasattr(self.model, "_last_original_obs"):
            vecnorm = self._find_vecnormalize()
            if vecnorm is not None:
                self.model._last_original_obs = vecnorm.unnormalize_obs(obs)


class CurriculumCallback(BaseCallback):
    """Ramps scenario difficulty as the validation success rate improves.

    Pairs with a :class:`PeriodicValidationCallback`: after each of that
    callback's validations, if the current stage's success target has held for
    ``advance_patience`` validations (and at least ``min_stage_steps`` have
    elapsed in the stage), step to the next, harder stage.  Must be placed
    *after* the validation callback in the callback list so its history is
    fresh.
    """

    def __init__(self, validation: "PeriodicValidationCallback",
                 cfg, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.validation = validation
        self.cfg = cfg                       # config.CurriculumCfg
        self.stage = 0
        self._stage_start_step = 0
        self._streak = 0
        self._seen_validations = 0

    # ------------------------------------------------------------------ #
    def _on_training_start(self) -> None:
        self._apply(0)

    def _apply(self, stage: int) -> None:
        stage = max(0, min(stage, len(self.cfg.stages) - 1))
        present, nonconflicting = self.cfg.stages[stage]
        self.training_env.env_method(
            "apply_curriculum_stage", stage, present, nonconflicting)
        self.stage = stage
        self._stage_start_step = self.num_timesteps
        self._streak = 0
        if self.verbose:
            LOGGER.info(
                "[curriculum] stage %d/%d | cyclist_present=%.2f "
                "nonconflicting=%.2f | @ step %d",
                stage, len(self.cfg.stages) - 1, present, nonconflicting,
                self.num_timesteps)

    def _on_step(self) -> bool:
        history = self.validation.history
        if len(history) == self._seen_validations:
            return True                       # no new validation since last check
        self._seen_validations = len(history)
        if self.stage >= len(self.cfg.stages) - 1:
            return True                       # already at the hardest stage

        success = float(history[-1].get("success", 0.0))
        self._streak = self._streak + 1 if success >= self.cfg.advance_success else 0
        elapsed = self.num_timesteps - self._stage_start_step
        if (self._streak >= self.cfg.advance_patience
                and elapsed >= self.cfg.min_stage_steps):
            self._apply(self.stage + 1)
        self.logger.record("curriculum/stage", self.stage)
        return True
