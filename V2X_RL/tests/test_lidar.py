"""Lidar clustering / static-object-rejection tests (no CARLA world needed)."""
import numpy as np
import pytest

from v2x_rl.config import LidarCfg
from v2x_rl.sensors.base import RangeRateTracker
from v2x_rl.sensors.lidar import LidarSensor


def make_lidar(cfg: LidarCfg | None = None) -> LidarSensor:
    cfg = cfg or LidarCfg()
    sensor = LidarSensor.__new__(LidarSensor)   # skip CARLA-touching __init__
    sensor.cfg = cfg
    sensor._points = np.zeros((0, 3), dtype=np.float32)
    sensor._tracker = RangeRateTracker(
        gate_m=4.0, lost_after_steps=cfg.track_lost_after_steps,
        ema_alpha=cfg.range_rate_ema_alpha)
    sensor._reset_static_state()
    return sensor


def _grid(x: float, y: float, half_w: float, half_l: float) -> np.ndarray:
    """A dense, jitter-free point box, so the centroid is exactly (x, y)."""
    gx, gy, gz = np.meshgrid(
        np.linspace(-half_l, half_l, 7),
        np.linspace(-half_w, half_w, 7),
        np.linspace(-1.1, 0.7, 6), indexing="ij")
    return np.column_stack([
        (x + gx).ravel(), (y + gy).ravel(), gz.ravel()]).astype(np.float32)


def pole(x: float, y: float) -> np.ndarray:      # thin footprint (0.16 m)
    return _grid(x, y, half_w=0.08, half_l=0.08)


def cyclist(x: float, y: float) -> np.ndarray:   # wider footprint (0.9 m)
    return _grid(x, y, half_w=0.30, half_l=0.45)


# --------------------------------------------------------------------------- #
def test_min_extent_gate_rejects_a_pole():
    lidar = make_lidar()
    lidar._points = pole(15.0, 0.0)
    assert lidar.clusters() == []


def test_cyclist_shaped_cluster_is_kept():
    lidar = make_lidar()
    lidar._points = cyclist(15.0, 0.0)
    clusters = lidar.clusters()
    assert len(clusters) == 1
    footprint = float(max(clusters[0][1][0], clusters[0][1][1]))
    cfg = LidarCfg()
    assert cfg.cluster_min_extent_m <= footprint <= cfg.cluster_max_extent_m


def test_static_cluster_is_dropped_then_its_spot_is_excluded():
    lidar = make_lidar()
    ego_xy, ego_yaw = np.zeros(2), 0.0
    valid = []
    for _ in range(12):
        lidar._points = cyclist(15.0, 0.0)          # never moves in the world
        valid.append(lidar.track(0.05, ego_xy, ego_yaw).valid)
    assert valid[0] is True                          # acquired at first
    assert valid[-1] is False                        # written off as static
    assert lidar._static_blobs                       # spot now excluded


def test_moving_cluster_stays_tracked():
    lidar = make_lidar()
    ego_xy, ego_yaw = np.zeros(2), 0.0
    track = None
    for k in range(15):
        lidar._points = cyclist(15.0, 0.30 * k)      # 6 m/s crossing
        track = lidar.track(0.05, ego_xy, ego_yaw)
    assert track.valid


def test_track_without_ego_pose_keeps_old_behaviour():
    lidar = make_lidar()
    lidar._points = cyclist(12.0, 0.0)
    track = lidar.track(0.05)                        # no static filtering
    assert track.valid
    assert track.range_m == pytest.approx(12.0, abs=1.0)
