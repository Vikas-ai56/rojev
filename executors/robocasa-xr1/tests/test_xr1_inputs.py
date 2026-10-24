"""The port of the checkpoint's inputs: the end-effector-first state from LeRobot's agent_pos, the history encoding
unpacked oldest first and padded, and the prompt with each camera's frames as one video."""
import numpy as np

from xr1_inputs import CAMERAS, LENGTH, STATE_DIM, messages, state14, unpack


def test_state14_reorders_agent_pos_end_effector_first_with_axis_angles():
    pos = np.array([1, 2, 3, 0, 0, 0, 1, 4, 5, 6, 0, 0, np.sin(0.25), np.cos(0.25), 7, 8], np.float32)
    s = state14(pos)
    assert s.shape == (14,) and s[:3].tolist() == [1, 2, 3] and np.allclose(s[3:6], 0)
    assert s[6:8].tolist() == [7, 8] and s[8:11].tolist() == [4, 5, 6] and np.allclose(s[11:14], [0, 0, 0.5])


def test_unpack_orders_frames_oldest_first_and_pads_the_state():
    images = {f"{cam}@{k}": np.full((4, 4, 3), k, np.uint8) for _, cam in CAMERAS for k in range(LENGTH)}
    frames, state = unpack(images, np.zeros(LENGTH * 16, np.float32))
    assert [int(f[0, 0, 0]) for f in frames["robot0_eye_in_hand"]] == list(range(LENGTH))
    assert state.shape == (1, LENGTH, STATE_DIM) and not state.any()
    prompt = messages(frames, "close the fridge")
    videos = [c for c in prompt[0]["content"] if c["type"] == "video"]
    assert len(videos) == 3 and all(len(v["video"]) == LENGTH for v in videos)
    assert prompt[0]["content"][-1]["text"].endswith("close the fridge /no_cot")
