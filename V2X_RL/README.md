# V2X_RL — learning to yield to a cyclist at an intersection

Reinforcement learning in CARLA for a car that turns left, turns right or goes
straight through a junction while a cyclist — which broadcasts its intended
path over a noisy, range-limited, one-way V2X link — may or may not be in the
way.

The agent controls **longitudinal motion only** (one continuous action: throttle
or brake). Steering is done by a pure-pursuit path follower, so the policy never
has to learn to steer. The objectives are to hold the target speed, yield when
the cyclist genuinely has priority, never collide, and never behave recklessly
(no bang-bang throttle, no braking for nothing).

```
                                      ▲ cyclist (broadcasts VAMs, never listens)
                                      │
   ego ──────────────────────────╮     │
   (depth cam + lidar,           ╰──►  ●  conflict point
    receives VAMs)                     │
```

## Contents

| Path | What it is |
| --- | --- |
| `v2x_rl/config.py` | Every tunable knob, as nested dataclasses; YAML-serialisable |
| `v2x_rl/geometry.py` | Pure-numpy path/conflict/TTC maths (no CARLA import) |
| `v2x_rl/scenario.py` | Junction discovery, left/right/straight routes, cyclist path, conflict point |
| `v2x_rl/controllers.py` | Ego lateral controller + kinematic cyclist controller |
| `v2x_rl/sensors/` | Depth camera → angular min-range sectors; lidar → DBSCAN clustering → track; noisy ground-truth ablation |
| `v2x_rl/v2x/` | ETSI-VAM-like message generation, wireless channel, receiver features |
| `v2x_rl/reward.py` | Reward function (pure, unit tested) |
| `v2x_rl/envs/` | The `gymnasium.Env` |
| `v2x_rl/callbacks.py` | Validation callback that saves **every** validation model |
| `v2x_rl/sb3_utils.py` | VecEnv construction + PPO/SAC/TD3 factory |
| `scripts/` | `inspect_map.py`, `train.py`, `evaluate.py`, `robustness_sweep.py` |
| `configs/` | `default.yaml` plus ablation profiles |
| `tests/` | 48 CARLA-free unit tests + CARLA integration tests |

## Setup

Uses the existing virtualenv:

```bash
VENV=~/Documents/VirtualEnvs/CarlaEnv/bin
$VENV/pip install gymnasium stable-baselines3 pytest    # already installed
```

CARLA 0.9.16 is expected at `~/Documents/CARLA_0.9.16`; set `CARLA_ROOT` if it
lives elsewhere. Start the simulator yourself:

```bash
~/Documents/CARLA_0.9.16/CarlaUE4.sh -quality-level=Low -RenderOffScreen
```

> **Hardware note.** This machine has an AMD GPU, so `torch.cuda.is_available()`
> is `False` and all training runs on CPU. That is fine for the default vector
> observations (a 256×256 MLP), but the `vector_depth` CNN mode will be slow.

## Quick start

```bash
cd V2X_RL
VENV=~/Documents/VirtualEnvs/CarlaEnv/bin

# 0. Unit tests (no simulator needed)
$VENV/python -m pytest tests -q

# 1. Find a junction and eyeball the three routes + the cyclist conflict.
#    Caches the result to cache/junctions.json.
$VENV/python scripts/inspect_map.py --list
$VENV/python scripts/inspect_map.py --episodes 8 --hold 6

# 2. Integration tests against the live simulator
$VENV/python -m pytest tests/test_env_carla.py -v

# 3. Train
$VENV/python scripts/train.py --algo sac --timesteps 300000 --val-freq 10000

# 4. Watch a saved checkpoint drive
$VENV/python scripts/evaluate.py runs/<run>/models/best_model.zip --render --episodes 10

# 5. How much does V2X actually help?
$VENV/python scripts/robustness_sweep.py runs/<run>/models/best_model.zip --sweep packet-loss
$VENV/python scripts/robustness_sweep.py runs/<run>/models/best_model.zip --sweep off
```

TensorBoard: `tensorboard --logdir runs`.

## The scenario

Each episode randomises:

* **manoeuvre** — `left`, `right` or `straight`, sampled uniformly. Routes are
  built by walking the lane graph and picking the branch at the junction whose
  probed heading change matches the requested manoeuvre, so nothing is
  hardcoded to specific spawn-point indices.
* **cyclist presence** — present with probability 0.75.
* **cyclist geometry** — the conflict depends on the manoeuvre:
  * *right turn* → cyclist goes straight in the kerb-side lane on the **same**
    approach (the classic right-hook);
  * *left turn* → cyclist comes from the **opposite** approach going straight
    (the oncoming conflict);
  * *straight* → oncoming cyclist going straight, i.e. **no** conflict, which
    is what teaches the agent not to brake needlessly.
  With probability `cyclist_nonconflicting_prob` even a turning episode gets a
  cyclist that turns away.
* **cyclist speed** (3–7 m/s) and **longitudinal offset** (±14 m), which is
  what decides who genuinely has priority. The offset is applied relative to
  the position that would make both arrive simultaneously, so the full
  spectrum from "clearly yield" to "clearly go" is covered.
* **ego initial speed** (20–40 km/h), so the policy never only sees standing
  starts.

The **conflict point** is the geometric crossing of the ego and cyclist path
polylines, computed once at reset. A crossing is only accepted if the two
headings differ by at least 20° — otherwise a cyclist riding along the adjacent
lane, permanently 2 m away, would register as a permanent conflict.

## Perception (no detector)

There is deliberately **no YOLO**. Two geometric backends, both on by default:

* **Depth camera → sector min-range.** The depth image is reduced to the
  nearest obstacle range (a robust 2nd percentile, not a raw minimum) in each
  of 8 angular slices of a horizontal band. Cheap and dense, but semantically
  blind — it cannot tell a cyclist from a wall.
* **Lidar → clustering → track.** ROI crop, ground/overhead removal, DBSCAN on
  the 2D projection, clusters gated by cyclist-plausible bounding-box extents,
  then nearest-cluster association across frames yielding range, bearing and a
  smoothed range-rate.

A third `groundtruth` backend (noisy, FOV-limited, occlusion-checked) exists as
the upper-bound ablation: `--gt-perception`.

Because a depth camera forces CARLA to render and a lidar does not,
`configs/lidar_only_fast.yaml` (`--no-rendering --no-depth`) is much faster for
long training runs.

## The V2X model

Modelled on **ETSI TS 103 300-3**, the VRU Awareness Message standard, whose
`vruMotionPredictionContainer` carries exactly what we need: `pathPrediction`,
`timeToCollision` and `trajectoryInterception`.

**Message** (`v2x/message.py`) — station id, VRU profile (bicyclist),
generation time, position, speed, heading, and the next *K* predicted path
points with per-point confidence.

**Generation** — event-triggered as the spec prescribes: a 10 Hz ceiling, a
1 Hz floor, and thresholds on position/speed/heading change in between. So the
message rate is not constant, which is realistic.

**Sender error** — the cyclist's own positioning error, not the channel's: an
Ornstein–Uhlenbeck (correlated, drifting) GNSS bias plus white noise, plus
noise on the path prediction that **grows with the horizon**.

**Channel** (`v2x/channel.py`):

| Effect | Model |
| --- | --- |
| range | hard cutoff at `max_range_m` |
| packet error rate | rises from `per_near` to `per_far` as `(d/R)^per_exponent` |
| NLOS | `nlos_extra_per` added when a CARLA raycast finds a building in the way |
| latency | uniform delay in `latency_ms`; messages can arrive out of order |

The NLOS term is the interesting one: at a junction, the case where V2X should
beat line-of-sight sensing is precisely when a building hides the cyclist — and
that is also when the radio link is worst.

**Receiver** (`v2x/receiver.py`) — keeps the freshest message (never
overwriting a newer one with a late arrival), expires anything older than
`max_message_age_s`, and derives the interception features by crossing the
*reported* path (extrapolated along its final heading, because a VAM prediction
horizon is shorter than the ego's decision horizon at 50 km/h) with the ego's
own route. Everything the policy sees is derived from received data only —
never from CARLA ground truth.

When nothing has been received the V2X features are **zeroed and a validity
flag drops to 0**, giving the network an explicit "I am blind" signal instead
of a silently stale value.

## Observation and action

**Action**: `Box(-1, 1, shape=(1,))`. Positive → throttle, negative → brake.
`action_rate_limit` (default 0.4/step) caps how fast the command may change,
modelling actuator dynamics and discouraging bang-bang control.

**Observation**, `obs_mode="vector"` (default) — 36 dims, all normalised and
clipped to [-1, 1]:

| Group | Dims | Contents |
| --- | --- | --- |
| ego | 12 | speed, target speed, speed error, previous throttle/brake, acceleration, steering command, distance to junction, distance to goal, manoeuvre one-hot(3) |
| depth | 8 | per-sector nearest range |
| lidar track | 5 | valid, range, bearing sin/cos, range-rate |
| V2X | 11 | valid, message age, range, bearing sin/cos, cyclist speed, cyclist distance to conflict, cyclist TTA, ego TTA, arrival gap, interception flag |

`obs_mode="vector_depth"` switches to `Dict{"vec", "depth"}` with a 64×64 uint8
depth image and a `MultiInputPolicy`. One flag, same code path.

Note the ego group contains distance to the **junction**, not to the conflict
point: the junction is known from the map, while the conflict point depends on
the cyclist and must be inferred from sensors and V2X.

## Reward

Ground truth is used for the reward (the environment knows everything); only
the *observation* is restricted to what the car could really sense.

| Term | Purpose |
| --- | --- |
| `speed` | track the target speed when no yield is required |
| `yield_correct` / `yield_violation` | reward waiting outside the conflict zone; penalise entering it before the cyclist clears, scaled by speed |
| `ttc` | quadratic penalty as time-to-collision drops below 3 s |
| `jerk`, `hard_brake` | comfort / anti-reckless |
| `unnecessary_brake`, `stalling` | **anti-overcautious** — only applied when no yield is required *and* no sensor sees anything close |
| `living_cost` | prefer finishing |
| `collision`, `goal`, `timeout` | terminal |

The last two rows matter most. Without them the optimal policy is "stop
forever, never collide"; `test_reward.py::test_holding_target_speed_beats_stopping_when_road_is_clear`
pins that down. Conversely, the `obstacle_perceived` guard means braking for
something the sensors genuinely see is never punished as unnecessary.

Yield priority is decided by comparing times-to-arrival: the cyclist has
priority while it has not cleared the conflict zone *and* would reach the
conflict no later than the ego plus a 1.5 s margin. A stopped ego has an
infinite TTA, so it is never forced to move into a conflict.

## Training and validation

`scripts/train.py --algo {ppo,sac,td3}`. **SAC is the default**: each CARLA step
is expensive and training is CPU-only, so sample efficiency dominates. PPO is
kept for the baseline. `gamma=0.995` (~10 s horizon) rather than the usual 0.99
(~5 s), which is too myopic for a 50 km/h approach.

`PeriodicValidationCallback` runs a deterministic validation phase every
`--val-freq` steps over a **fixed set of seeds**, and:

* logs mean return, success/collision/timeout rate, yield-correctness rate,
  mean speed error, mean jerk, unnecessary-brake fraction, min TTC, min cyclist
  distance, V2X availability and loss rate to TensorBoard;
* **saves every validation model**, not just the best one:
  `runs/<run>/models/val_step<N>_r<return>_col<rate>_suc<rate>.zip` together
  with the matching `*_vecnormalize.pkl`;
* appends a row to `runs/<run>/models/validation_log.csv`;
* also maintains `best_model.zip` / `best_vecnormalize.pkl`.

Any of those snapshots can be replayed later with `scripts/evaluate.py`.

> **A caveat, stated plainly.** Validation runs on the *training* environment.
> A CARLA server hosts one world, and spawning a second ego vehicle in it would
> let the two environments interfere, so a separate eval env is not an option
> here. The consequence is that the training episode in progress is abandoned at
> each validation; the learner's cached observation is re-synchronised
> afterwards. With 30-second episodes this costs at most one episode per
> validation. If you ever run a second CARLA server, a genuinely separate eval
> env would be the cleaner fix.

## Ablations worth running

```bash
# Does V2X help at all?
python scripts/train.py --algo sac --run-name sac_v2x
python scripts/train.py --algo sac --no-v2x --run-name sac_no_v2x

# Is perception the bottleneck?
python scripts/train.py --algo sac --config configs/gt_perception.yaml

# How gracefully does a V2X-trained policy degrade?
python scripts/robustness_sweep.py runs/sac_v2x/models/best_model.zip --sweep packet-loss
python scripts/robustness_sweep.py runs/sac_v2x/models/best_model.zip --sweep nlos
```

## Configuration

Anything in `config.py` can be set from a YAML file or overridden on the command
line with dotted keys:

```bash
python scripts/train.py --config configs/lidar_only_fast.yaml \
    --set v2x.per_far=0.6 scenario.target_speed_kmh=40 scenario.cyclist_present_prob=1.0
```

The exact config of every run is written to `runs/<run>/env_config.yaml`, and
`evaluate.py` picks it up automatically so evaluation always matches training.
