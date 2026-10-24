"""The 2D "placed inside" geometry (an object within an open container's outline counts as inside, one beside it
does not; the prompt point lies on the object) and the live frame recorder (images and states in capture order)."""
import numpy as np

from harness.tracking import central_pixel, filled, inside_share
from sam_check import decide


def box(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_an_open_container_outline_is_filled_so_its_interior_counts():
    walls = box(20, 20, 2, 18, 2, 18) & ~box(20, 20, 4, 16, 4, 16)
    assert filled(walls)[10, 10]
    assert not filled(walls)[0, 0]


def test_an_object_inside_the_outline_is_inside_and_one_beside_it_is_not():
    walls = box(20, 30, 2, 18, 2, 18) & ~box(20, 30, 4, 16, 4, 16)
    assert inside_share(box(20, 30, 8, 12, 8, 12), walls) == 1.0
    assert inside_share(box(20, 30, 8, 12, 22, 26), walls) == 0.0


def test_the_central_pixel_lies_on_a_concave_object():
    ring = box(21, 21, 0, 21, 0, 21) & ~box(21, 21, 5, 16, 5, 16)
    x, y = central_pixel(ring)
    assert ring[int(y), int(x)]


def test_no_camera_seeing_the_object_gives_no_answer():
    assert decide([None, None]) is None
    assert decide([0.2, None, 0.6]) is True


def test_the_frame_recorder_saves_images_and_simulator_states_in_capture_order(tmp_path):
    from types import SimpleNamespace

    from harness.tracking import FrameRecorder

    captures = iter(SimpleNamespace(t=float(i), obs_id=i, observation=SimpleNamespace(
        images={"a": np.full((2, 2, 3), i, np.uint8), "b": np.zeros((2, 2, 3), np.uint8)})) for i in range(3))
    states = iter(np.array([float(i), -float(i)]) for i in range(3))
    recorder = FrameRecorder(lambda: next(captures), lambda fn: next(states), ["a", "b"], tmp_path / "frames")
    for _ in range(3):
        recorder.record_once()
    recorder.save()
    data = np.load(tmp_path / "frames" / "frames.npz")
    assert list(data["t"]) == [0.0, 1.0, 2.0]
    assert data["images"].shape == (3, 2, 2, 2, 3)
    assert data["images"][2, 0, 0, 0, 0] == 2
    assert list(data["state"][1]) == [1.0, -1.0]
    assert list(data["cameras"]) == ["a", "b"]
