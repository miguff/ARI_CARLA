import numpy as np
import pytest

from v2x_rl.config import V2XCfg
from v2x_rl.v2x import V2XChannel, V2XReceiver, VAMGenerator


def straight_path(start_x=0.0, n=8, dx=2.0, y=0.0):
    return np.stack([[start_x + i * dx, y] for i in range(1, n + 1)])


# --------------------------------------------------------------------------- #
#  Generation
# --------------------------------------------------------------------------- #
def test_generator_respects_max_rate():
    cfg = V2XCfg(max_rate_hz=10.0, min_rate_hz=1.0, trigger_position_delta_m=4.0)
    gen = VAMGenerator(cfg, np.random.default_rng(0))
    gen.reset()

    assert gen.generate(0.0, np.zeros(2), 5.0, 0.0, straight_path()) is not None
    # 50 ms later the cyclist has moved far enough to trigger, but the 10 Hz
    # ceiling still suppresses the message.
    assert gen.generate(0.05, np.array([9.0, 0.0]), 5.0, 0.0, straight_path()) is None
    # At 100 ms the ceiling allows it and the position delta triggers it.
    assert gen.generate(0.10, np.array([9.0, 0.0]), 5.0, 0.0, straight_path()) is not None


def test_generator_suppresses_messages_when_nothing_changes():
    """Event-triggered generation: small changes do not warrant a message."""
    cfg = V2XCfg(max_rate_hz=10.0, min_rate_hz=1.0, trigger_position_delta_m=4.0,
                 trigger_speed_delta_ms=0.5, trigger_heading_delta_deg=4.0)
    gen = VAMGenerator(cfg, np.random.default_rng(0))
    gen.reset()
    assert gen.generate(0.0, np.zeros(2), 5.0, 0.0, straight_path()) is not None
    # Well past the 10 Hz ceiling, but only 0.5 m travelled and no change in
    # speed or heading -> suppressed until the 1 Hz floor.
    assert gen.generate(0.20, np.array([0.5, 0.0]), 5.0, 0.0, straight_path()) is None


def test_generator_rate_floor_fires_without_change():
    cfg = V2XCfg(max_rate_hz=10.0, min_rate_hz=1.0,
                 trigger_position_delta_m=1e6,
                 trigger_speed_delta_ms=1e6,
                 trigger_heading_delta_deg=1e6)
    gen = VAMGenerator(cfg, np.random.default_rng(0))
    gen.reset()
    assert gen.generate(0.0, np.zeros(2), 0.0, 0.0, straight_path()) is not None
    # Nothing changed, so only the 1 Hz floor can trigger the next message.
    assert gen.generate(0.5, np.zeros(2), 0.0, 0.0, straight_path()) is None
    assert gen.generate(1.0, np.zeros(2), 0.0, 0.0, straight_path()) is not None


def test_generator_heading_change_triggers():
    cfg = V2XCfg(max_rate_hz=10.0, min_rate_hz=1.0,
                 trigger_position_delta_m=1e6, trigger_speed_delta_ms=1e6,
                 trigger_heading_delta_deg=4.0, heading_noise_std_deg=0.0)
    gen = VAMGenerator(cfg, np.random.default_rng(0))
    gen.reset()
    gen.generate(0.0, np.zeros(2), 5.0, 0.0, straight_path())
    assert gen.generate(0.2, np.zeros(2), 5.0, 1.0, straight_path()) is None
    assert gen.generate(0.4, np.zeros(2), 5.0, 20.0, straight_path()) is not None


def test_generator_noise_grows_along_prediction_horizon():
    cfg = V2XCfg(gnss_bias_std_m=0.0, gnss_white_std_m=0.0,
                 path_prediction_noise_std_m_per_s=1.0,
                 path_prediction_points=6, path_prediction_dt_s=0.5)
    rng = np.random.default_rng(1)
    gen = VAMGenerator(cfg, rng)
    gen.reset()
    message = gen.generate(0.0, np.zeros(2), 5.0, 0.0, straight_path())
    assert message is not None
    # Confidence must increase monotonically with the horizon.
    assert np.all(np.diff(message.path_confidence_m) > 0)
    assert len(message.path_prediction) == 6


def test_generator_gnss_bias_is_correlated():
    """Consecutive messages should share most of their position error."""
    cfg = V2XCfg(gnss_bias_std_m=2.0, gnss_bias_tau_s=100.0, gnss_white_std_m=0.0,
                 max_rate_hz=10.0)
    gen = VAMGenerator(cfg, np.random.default_rng(3))
    gen.reset()
    errors = []
    for i in range(20):
        message = gen.generate(i * 0.1, np.array([i * 0.5, 0.0]), 5.0, 0.0,
                               straight_path())
        if message is not None:
            errors.append(message.position - np.array([i * 0.5, 0.0]))
    errors = np.array(errors)
    # A random walk with tau=100 s barely moves over 2 s.
    assert np.std(np.diff(errors[:, 0])) < 0.5
    assert np.abs(errors[:, 0]).mean() > 0.1


# --------------------------------------------------------------------------- #
#  Channel
# --------------------------------------------------------------------------- #
def test_packet_error_rate_monotonic_and_bounded():
    cfg = V2XCfg(max_range_m=100.0, per_near=0.02, per_far=0.9, per_exponent=2.0,
                 nlos_extra_per=0.5)
    channel = V2XChannel(cfg, np.random.default_rng(0))
    rates = [channel.packet_error_rate(d, True) for d in (0, 25, 50, 75, 100)]
    assert rates[0] == pytest.approx(0.02)
    assert rates[-1] == pytest.approx(0.9)
    assert all(b >= a for a, b in zip(rates, rates[1:]))
    assert channel.packet_error_rate(150.0, True) == 1.0
    # Blocked line of sight is strictly worse, and still clipped to 1.
    assert channel.packet_error_rate(50.0, False) > channel.packet_error_rate(50.0, True)
    assert channel.packet_error_rate(99.0, False) <= 1.0


def test_channel_loss_rate_matches_configured_per():
    # trigger_position_delta_m=0 makes the generator fire at the 10 Hz ceiling,
    # giving one transmission per step so the loss statistics are clean.
    cfg = V2XCfg(max_range_m=100.0, per_near=0.3, per_far=0.3, per_exponent=1.0,
                 latency_ms=(0.0, 0.0), trigger_position_delta_m=0.0)
    channel = V2XChannel(cfg, np.random.default_rng(7))
    gen = VAMGenerator(cfg, np.random.default_rng(7))
    gen.reset()
    steps = 4000
    for i in range(steps):
        message = gen.generate(i * 0.1, np.array([i * 0.5, 0.0]), 5.0, 0.0,
                               straight_path())
        channel.transmit(message, i * 0.1, 50.0, True)
        channel.poll(i * 0.1)
    assert channel.stats.sent == steps
    assert channel.stats.loss_rate == pytest.approx(0.3, abs=0.03)
    assert channel.stats.dropped_range == 0


def test_channel_range_cutoff():
    cfg = V2XCfg(max_range_m=50.0)
    channel = V2XChannel(cfg, np.random.default_rng(0))
    gen = VAMGenerator(cfg, np.random.default_rng(0))
    gen.reset()
    message = gen.generate(0.0, np.zeros(2), 5.0, 0.0, straight_path())
    channel.transmit(message, 0.0, 80.0, True)
    assert channel.stats.dropped_range == 1
    assert channel.poll(10.0) == []


def test_channel_latency_delays_delivery():
    cfg = V2XCfg(per_near=0.0, per_far=0.0, nlos_extra_per=0.0,
                 latency_ms=(100.0, 100.0))
    channel = V2XChannel(cfg, np.random.default_rng(0))
    gen = VAMGenerator(cfg, np.random.default_rng(0))
    gen.reset()
    message = gen.generate(0.0, np.zeros(2), 5.0, 0.0, straight_path())
    channel.transmit(message, 0.0, 10.0, True)
    assert channel.poll(0.05) == []
    assert len(channel.poll(0.10)) == 1


def test_channel_disabled_transmits_nothing():
    cfg = V2XCfg(enabled=False)
    channel = V2XChannel(cfg, np.random.default_rng(0))
    gen = VAMGenerator(cfg, np.random.default_rng(0))
    gen.reset()
    channel.transmit(gen.generate(0.0, np.zeros(2), 5.0, 0.0, straight_path()),
                     0.0, 5.0, True)
    assert channel.stats.sent == 0
    assert channel.poll(100.0) == []


# --------------------------------------------------------------------------- #
#  Receiver
# --------------------------------------------------------------------------- #
def _message(cfg, now, position, heading, path, speed=5.0):
    gen = VAMGenerator(cfg, np.random.default_rng(0))
    gen.reset()
    return gen.generate(now, np.asarray(position, dtype=float), speed, heading, path)


def test_receiver_invalid_without_messages():
    cfg = V2XCfg()
    receiver = V2XReceiver(cfg)
    derived = receiver.derive(1.0, np.zeros(2), 0.0, 10.0,
                              np.array([[0.0, 0.0], [50.0, 0.0]]))
    assert not derived.valid
    assert np.all(receiver.features(derived) == 0.0)


def test_receiver_expires_stale_messages():
    cfg = V2XCfg(max_message_age_s=1.0, gnss_bias_std_m=0.0, gnss_white_std_m=0.0)
    receiver = V2XReceiver(cfg)
    receiver.update([_message(cfg, 0.0, (0.0, -20.0), 90.0, straight_path())])
    assert receiver.is_valid(0.5)
    assert not receiver.is_valid(1.5)


def test_receiver_keeps_newest_on_out_of_order_arrival():
    cfg = V2XCfg(gnss_bias_std_m=0.0, gnss_white_std_m=0.0,
                 path_prediction_noise_std_m_per_s=0.0)
    receiver = V2XReceiver(cfg)
    old = _message(cfg, 1.0, (0.0, -20.0), 90.0, straight_path())
    new = _message(cfg, 2.0, (0.0, -10.0), 90.0, straight_path())
    receiver.update([new, old])
    assert receiver.latest.generation_time_s == 2.0


def test_receiver_detects_trajectory_interception():
    """A cyclist crossing the ego route must produce an interception + gap."""
    cfg = V2XCfg(gnss_bias_std_m=0.0, gnss_white_std_m=0.0,
                 path_prediction_noise_std_m_per_s=0.0,
                 speed_noise_std_ms=0.0, heading_noise_std_deg=0.0,
                 path_prediction_points=6, path_prediction_dt_s=0.5,
                 receiver_extrapolation_m=40.0)
    receiver = V2XReceiver(cfg)
    # Cyclist at (30, -20) heading +y at 5 m/s: crosses the ego's +x route.
    cyclist_path = np.stack([[30.0, -20.0 + 2.5 * i] for i in range(1, 7)])
    receiver.update([_message(cfg, 0.0, (30.0, -20.0), 90.0, cyclist_path)])

    ego_ahead = np.array([[0.0, 0.0], [60.0, 0.0]])
    derived = receiver.derive(0.0, np.zeros(2), 0.0, 10.0, ego_ahead)

    assert derived.valid and derived.interception
    assert derived.conflict_xy[0] == pytest.approx(30.0, abs=1.0)
    assert derived.ego_dist_to_conflict_m == pytest.approx(30.0, abs=1.0)
    assert derived.cyclist_dist_to_conflict_m == pytest.approx(20.0, abs=2.0)
    # Ego 3 s away, cyclist 4 s away -> the ego arrives first, gap negative.
    assert derived.ego_tta_s == pytest.approx(3.0, abs=0.2)
    assert derived.cyclist_tta_s == pytest.approx(4.0, abs=0.4)
    assert derived.arrival_gap_s < 0.0


def test_receiver_no_interception_for_parallel_cyclist():
    cfg = V2XCfg(gnss_bias_std_m=0.0, gnss_white_std_m=0.0,
                 path_prediction_noise_std_m_per_s=0.0)
    receiver = V2XReceiver(cfg)
    # Cyclist riding parallel to the ego, 8 m to the side.
    cyclist_path = np.stack([[10.0 + 2.5 * i, 8.0] for i in range(1, 7)])
    receiver.update([_message(cfg, 0.0, (10.0, 8.0), 0.0, cyclist_path)])
    derived = receiver.derive(0.0, np.zeros(2), 0.0, 10.0,
                              np.array([[0.0, 0.0], [60.0, 0.0]]))
    assert derived.valid and not derived.interception
    assert derived.range_m == pytest.approx(np.hypot(10.0, 8.0), abs=0.1)


def test_receiver_features_are_bounded_and_named():
    from v2x_rl.v2x import V2X_FEATURE_DIM, V2X_FEATURE_NAMES
    cfg = V2XCfg()
    receiver = V2XReceiver(cfg)
    cyclist_path = np.stack([[30.0, -20.0 + 2.5 * i] for i in range(1, 7)])
    receiver.update([_message(cfg, 0.0, (30.0, -20.0), 90.0, cyclist_path)])
    features = receiver.features(
        receiver.derive(0.0, np.zeros(2), 0.0, 10.0,
                        np.array([[0.0, 0.0], [60.0, 0.0]])))
    assert features.shape == (V2X_FEATURE_DIM,)
    assert len(V2X_FEATURE_NAMES) == V2X_FEATURE_DIM
    assert np.all(np.abs(features) <= 1.0 + 1e-6)
    assert features.dtype == np.float32


# --------------------------------------------------------------------------- #
#  Gap filling
# --------------------------------------------------------------------------- #
def test_gap_fill_defaults_to_none_and_changes_nothing():
    """The whole point of the CLI/config default: opt-in only."""
    from v2x_rl.v2x import build_gap_filler
    assert V2XCfg().gap_fill == "none"
    assert build_gap_filler(V2XCfg()) is None


def test_build_gap_filler_rejects_unknown_mode():
    from v2x_rl.v2x import build_gap_filler
    with pytest.raises(ValueError):
        build_gap_filler(V2XCfg(gap_fill="particle_filter"))


def test_receiver_without_filler_still_goes_blind_past_max_age():
    """Unchanged baseline behaviour: no filler means no second chance."""
    cfg = V2XCfg(max_message_age_s=1.0, gap_fill="none",
                gnss_bias_std_m=0.0, gnss_white_std_m=0.0)
    receiver = V2XReceiver(cfg)
    receiver.update([_message(cfg, 0.0, (10.0, 0.0), 0.0, straight_path())])
    derived = receiver.derive(1.5, np.zeros(2), 0.0, 10.0,
                              np.array([[0.0, 0.0], [60.0, 0.0]]))
    assert not derived.valid


@pytest.mark.parametrize("mode", ["dead_reckoning", "kalman"])
def test_gap_fill_bridges_a_short_drop_out(mode):
    """Past max_message_age_s but inside gap_fill_max_age_s: still valid."""
    cfg = V2XCfg(max_message_age_s=1.0, gap_fill_max_age_s=4.0, gap_fill=mode,
                gnss_bias_std_m=0.0, gnss_white_std_m=0.0,
                path_prediction_noise_std_m_per_s=0.0,
                speed_noise_std_ms=0.0, heading_noise_std_deg=0.0)
    receiver = V2XReceiver(cfg)
    # Cyclist moving in +x at 5 m/s, starting at (0, -20).
    cyclist_path = np.stack([[2.5 * i, -20.0] for i in range(1, 7)])
    receiver.update([_message(cfg, 0.0, (0.0, -20.0), 0.0, cyclist_path, speed=5.0)])

    derived = receiver.derive(2.0, np.zeros(2), 0.0, 10.0,
                              np.array([[0.0, 0.0], [60.0, 0.0]]))
    assert derived.valid
    # 2 s at 5 m/s from x=0 -> the filler's extrapolated position is ~10 m out.
    assert derived.range_m == pytest.approx(np.hypot(10.0, 20.0), abs=1.5)
    features = receiver.features(derived)
    assert np.all(np.abs(features) <= 1.0 + 1e-6)


@pytest.mark.parametrize("mode", ["dead_reckoning", "kalman"])
def test_gap_fill_still_expires_past_its_own_max_age(mode):
    """A filler is a longer leash, not an indefinite one."""
    cfg = V2XCfg(max_message_age_s=1.0, gap_fill_max_age_s=3.0, gap_fill=mode,
                gnss_bias_std_m=0.0, gnss_white_std_m=0.0)
    receiver = V2XReceiver(cfg)
    receiver.update([_message(cfg, 0.0, (10.0, 0.0), 0.0, straight_path())])
    derived = receiver.derive(5.0, np.zeros(2), 0.0, 10.0,
                              np.array([[0.0, 0.0], [60.0, 0.0]]))
    assert not derived.valid
    assert np.all(receiver.features(derived) == 0.0)


def test_dead_reckoning_extrapolates_at_constant_velocity():
    from v2x_rl.v2x import DeadReckoningFiller
    cfg = V2XCfg(gap_fill_max_age_s=10.0, gnss_bias_std_m=0.0, gnss_white_std_m=0.0,
                speed_noise_std_ms=0.0, heading_noise_std_deg=0.0)
    filler = DeadReckoningFiller(cfg)
    message = _message(cfg, 0.0, (0.0, 0.0), 0.0, straight_path(), speed=5.0)
    filler.observe(message)
    position, speed_ms, heading_deg = filler.predict(2.0)
    assert position == pytest.approx([10.0, 0.0], abs=0.1)
    assert speed_ms == pytest.approx(5.0, abs=0.1)
    assert heading_deg == pytest.approx(0.0, abs=1.0)


def test_kalman_filler_smooths_noisy_measurements():
    """Repeated noisy fixes of a stationary point should average toward it.

    Builds VAMs directly rather than through VAMGenerator: the generator's
    own GNSS bias/noise model would add an extra, KF-invisible offset on top
    of the synthetic per-sample noise this test controls, which is exactly
    the kind of unmodelled bias a real Kalman filter can't average away --
    not what this test is checking.
    """
    from v2x_rl.v2x import VAM, KalmanFiller
    cfg = V2XCfg(gap_fill_max_age_s=10.0, kf_process_accel_std_ms2=0.01,
                gnss_white_std_m=1.0)
    filler = KalmanFiller(cfg)
    rng = np.random.default_rng(0)
    truth = np.array([20.0, 5.0])
    t = 0.0
    for _ in range(50):
        t += 0.2
        noisy = truth + rng.normal(0.0, cfg.gnss_white_std_m, size=2)
        filler.observe(VAM(station_id=1, vru_profile=2, generation_time_s=t,
                           position=noisy, speed_ms=0.0, heading_deg=0.0,
                           path_prediction=np.zeros((0, 2)),
                           path_prediction_dt_s=0.5,
                           path_confidence_m=np.zeros(0)))
    position, _, _ = filler.predict(t)
    # The filtered estimate should land much closer to the truth than a
    # single raw 1-sigma-noise measurement would.
    assert np.linalg.norm(position - truth) < 0.7


def test_kalman_filler_expires_like_dead_reckoning():
    from v2x_rl.v2x import KalmanFiller
    cfg = V2XCfg(gap_fill_max_age_s=2.0)
    filler = KalmanFiller(cfg)
    filler.observe(_message(cfg, 0.0, (0.0, 0.0), 0.0, straight_path(), speed=5.0))
    assert filler.predict(1.0) is not None
    assert filler.predict(3.0) is None
