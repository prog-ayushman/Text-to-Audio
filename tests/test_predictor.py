import numpy as np
import pytest

from predictor import MusicPredictor, audio_stats


@pytest.mark.parametrize("bad", ["", "   ", 123, None])
def test_bad_prompt_rejected(bad):
    with pytest.raises(ValueError):
        MusicPredictor.validate(bad, 5, 0)


def test_overlong_prompt_rejected():
    with pytest.raises(ValueError):
        MusicPredictor.validate("a" * 301, 5, 0)


@pytest.mark.parametrize("bad", [0.5, 11, True, "5"])
def test_bad_duration_rejected(bad):
    with pytest.raises(ValueError):
        MusicPredictor.validate("piano", bad, 0)


@pytest.mark.parametrize("bad", [1.5, True, "0"])
def test_bad_seed_rejected(bad):
    with pytest.raises(ValueError):
        MusicPredictor.validate("piano", 5, bad)


def test_valid_input_returns_stripped_prompt():
    assert MusicPredictor.validate("  calm piano  ", 5, 0) == "calm piano"


def test_audio_stats_on_sine_wave():
    rate = 32000
    t = np.arange(rate) / rate
    s = audio_stats((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), rate)
    assert s["peak"] == pytest.approx(0.5, abs=1e-3)
    assert s["rms"] == pytest.approx(0.5 / np.sqrt(2), abs=1e-3)
    assert s["silence_fraction"] == 0.0
    assert s["all_finite"] is True


def test_audio_stats_silence_and_nan():
    rate = 32000
    assert audio_stats(np.zeros(rate, dtype=np.float32), rate)["silence_fraction"] == 1.0
    bad = np.zeros(rate, dtype=np.float32)
    bad[10] = np.nan
    assert audio_stats(bad, rate)["all_finite"] is False


def test_revision_pinned_and_overridable():
    assert MusicPredictor().revision == MusicPredictor.DEFAULT_REVISION
    assert MusicPredictor(revision="abc").revision == "abc"