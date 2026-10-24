"""Corrupted camera frames are caught and make a run unscorable: noise and blank images are classified, real-looking
ones are not; the check reports the first bad frame; the run invariants flag it."""
from types import SimpleNamespace

import numpy as np

from harness.frame_check import FrameCheck, frame_problem
from harness.invariants import frames_real


def scene(seed=0):
    """A smooth image with mild texture, like a rendered kitchen (neighbour difference ~2, standard deviation ~40)."""
    y, x = np.mgrid[0:64, 0:64]
    base = (x * 2 + y)[..., None] + np.random.default_rng(seed).integers(0, 4, (64, 64, 3))
    return np.clip(base, 0, 255).astype(np.uint8)


def test_noise_and_blank_frames_are_caught_and_a_scene_is_not():
    assert frame_problem(scene()) is None
    assert frame_problem(np.random.default_rng(1).integers(0, 255, (64, 64, 3)).astype(np.uint8))[0] == "noise"
    assert frame_problem(np.zeros((64, 64, 3), np.uint8))[0] == "blank"


def test_the_check_counts_bad_frames_and_keeps_the_first():
    frames = iter([{"a": scene(), "b": scene(1)}, {"a": scene(), "b": np.zeros((64, 64, 3), np.uint8)}])
    captures = iter(SimpleNamespace(t=float(i), observation=SimpleNamespace(images=next(frames))) for i in range(2))
    check = FrameCheck(lambda: next(captures), ["a", "b"])
    check.check_once()
    check.check_once()
    report = check.report()
    assert report["checked"] == 2 and report["bad"] == {"a": 0, "b": 1} and not report["valid"]
    assert report["first_bad"]["camera"] == "b" and report["first_bad"]["problem"] == "blank"


def test_a_run_with_bad_frames_breaks_the_invariant_and_one_without_the_check_does_not():
    bad = SimpleNamespace(summary={"frame_check": {"valid": False, "bad": {"b": 1}, "first_bad": {"t": 1.0}}})
    assert frames_real(bad) and not frames_real(SimpleNamespace(summary={}))
    assert not frames_real(SimpleNamespace(summary={"frame_check": {"valid": True}}))
