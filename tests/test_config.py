"""Config validation and live apply_config."""

import pytest
from pydantic import ValidationError

from config import AppConfig
from engine.state_machine import PiracyStateMachine


def test_rejects_invalid_nir_mode_and_speed() -> None:
    with pytest.raises(ValidationError):
        AppConfig(_env_file=None, nir_mode="banana")
    with pytest.raises(ValidationError):
        AppConfig(_env_file=None, playback_speed=8)
    with pytest.raises(ValidationError):
        AppConfig(_env_file=None, playback_speed=0.1)


def test_accepts_simulated_mode_and_hides_token() -> None:
    config = AppConfig(_env_file=None, nir_mode="SIMULATED", api_token="secret", playback_speed=1.0)
    assert config.nir_mode == "simulated"
    public = config.public_dict()
    assert public["auth_enabled"] is True
    assert "api_token" not in public
    assert "secret" not in str(public)


def test_apply_config_pushes_threshold_and_rejects_bad_merge() -> None:
    from app import CommandCenterRuntime, ModelBundle

    class _Pose:
        def __init__(self) -> None:
            self.confidence_threshold = 0.5

        def set_confidence_threshold(self, value: float) -> None:
            self.confidence_threshold = value

    config = AppConfig(_env_file=None, conf_threshold=0.5, piracy_threshold_sec=30, target_fps=10)
    runtime = CommandCenterRuntime(config=config, autoload_models=False)
    runtime.models = ModelBundle.unavailable("test")
    runtime.models.pose = _Pose()
    machine = PiracyStateMachine(30.0)
    runtime._states[1] = machine

    updated = runtime.apply_config({"conf_threshold": 0.4, "piracy_threshold_sec": 12, "target_fps": 8})
    assert updated["conf_threshold"] == pytest.approx(0.4)
    assert runtime.models.pose.confidence_threshold == pytest.approx(0.4)
    assert machine.piracy_threshold_sec == pytest.approx(12)
    assert runtime.source.target_fps == 8

    with pytest.raises(ValidationError):
        runtime.apply_config({"playback_speed": 9})
    assert runtime.config.playback_speed == pytest.approx(1.0)
