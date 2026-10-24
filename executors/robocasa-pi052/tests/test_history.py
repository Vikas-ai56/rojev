"""History windows match the model's own evaluation (clamped at the start, interval apart, ending at the asked
observation, including one not filed yet), and a history request carries every frame and state in order."""
import numpy as np

from history import FrameHistory, HistoryPolicy
from rojev.protocol import Observation
from rojev.system1 import ActionChunk, Policy


def obs(i):
    return Observation({"cam": np.full((2, 2, 3), i, np.uint8)}, np.array([float(i), -float(i)], np.float32), "t")


def test_window_is_interval_apart_clamped_at_the_start_and_ends_at_the_asked_observation():
    h = FrameHistory(7)
    frames = [obs(i) for i in range(10)]
    h.add(frames[0])
    assert [int(f.state[0]) for f in h.window(frames[0], 4, 2)] == [0, 0, 0, 0]
    for f in frames[1:8]:
        h.add(f)
    assert [int(f.state[0]) for f in h.window(frames[7], 4, 2)] == [1, 3, 5, 7]
    assert [int(f.state[0]) for f in h.window(frames[5], 4, 2)] == [1, 1, 3, 5]
    assert [int(f.state[0]) for f in h.window(frames[8], 4, 2)] == [2, 4, 6, 8]


def test_a_history_request_carries_frames_oldest_first_and_maps_the_actions():
    seen = {}

    class Inner(Policy):
        def act(self, o, subtask, timeout_s=None):
            seen.update(images=sorted(o.images), state=o.state.tolist())
            return ActionChunk(np.ones((3, 2), np.float32))

        def reset(self):
            pass

    h = FrameHistory(7)
    frames = [obs(i) for i in range(4)]
    for f in frames:
        h.add(f)
    chunk = HistoryPolicy(Inner(), h, 2, 2, lambda a: a * 2).act(frames[3], "")
    assert seen == {"images": ["cam@0", "cam@1"], "state": [1.0, -1.0, 3.0, -3.0]}
    assert np.all(chunk.actions == 2)
