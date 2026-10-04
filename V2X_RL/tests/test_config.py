import pytest

from v2x_rl.config import EnvCfg


def test_roundtrip_through_dict():
    cfg = EnvCfg()
    restored = EnvCfg.from_dict(cfg.to_dict())
    assert restored.to_dict() == cfg.to_dict()


def test_save_and_load(tmp_path):
    cfg = EnvCfg()
    cfg.v2x.per_far = 0.55
    path = tmp_path / "cfg.yaml"
    cfg.save(str(path))
    assert EnvCfg.load(str(path)).v2x.per_far == pytest.approx(0.55)


def test_merge_dotted_overrides():
    cfg = EnvCfg().merge({
        "v2x.per_far": 1.0,
        "scenario.target_speed_kmh": 30.0,
        "obs_mode": "vector_depth",
    })
    assert cfg.v2x.per_far == 1.0
    assert cfg.scenario.target_speed_kmh == 30.0
    assert cfg.obs_mode == "vector_depth"
    # Untouched values keep their defaults.
    assert cfg.v2x.per_near == EnvCfg().v2x.per_near


def test_merge_does_not_mutate_the_original():
    cfg = EnvCfg()
    cfg.merge({"v2x.enabled": False})
    assert cfg.v2x.enabled is True


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown config keys"):
        EnvCfg.from_dict({"nope": 1})
    with pytest.raises(ValueError, match="unknown config keys"):
        EnvCfg.from_dict({"v2x": {"per_infinity": 1.0}})
