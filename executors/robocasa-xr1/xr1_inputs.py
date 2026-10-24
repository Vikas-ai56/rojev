"""Xiaomi-Robotics-1 (RoboCasa365 checkpoint) inputs, ported from its evaluation client (Xiaomi-Robotics-1
`eval_robocasa365/entry.py`, Apache-2.0) so a query here matches the published evaluation exactly.

A query is `LENGTH` frames per camera, `INTERVAL` control steps apart, ending at the newest; each frame center-cropped
by `CROP` and resized back; a 14-value state per frame (end effector first) zero-padded to `STATE_DIM`; the
instruction inside the fixed prompt. The robot side sends the frames as `<camera>@<k>` images (k = 0 oldest) and the
states as one stacked vector of LeRobot RoboCasa `agent_pos` rows (16 values each).
"""
import numpy as np
from PIL import Image

ROBOT_TYPE = "robocasa365"
LENGTH, INTERVAL = 4, 2
CROP = 0.95
STATE_DIM = 60
ACTION_DIM = 12
CAMERAS = (("Left camera: ", "robot0_agentview_left"), ("\nRight camera: ", "robot0_agentview_right"),
           ("\nWrist camera: ", "robot0_eye_in_hand"))
AGENT_POS = 16


def quat_xyzw_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    xyz = quaternion[:3]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[3], -1.0, 1.0))
    return (xyz / sin_half * angle).astype(np.float32)


def state14(agent_pos: np.ndarray) -> np.ndarray:
    """LeRobot RoboCasa `agent_pos` (end-effector position 0:3 and quaternion 3:7 relative to the base, base position
    7:10 and quaternion 10:14, gripper 14:16; quaternions xyzw; `lerobot/envs/robocasa.py`) → the checkpoint's
    end-effector-first state: ee position, ee axis-angle, gripper, base position, base axis-angle."""
    p = np.asarray(agent_pos, dtype=np.float32).reshape(-1)
    return np.concatenate([p[0:3], quat_xyzw_to_axis_angle(p[3:7]), p[14:16], p[7:10],
                           quat_xyzw_to_axis_angle(p[10:14])]).astype(np.float32)


def center_crop(image: np.ndarray, crop_ratio: float = CROP) -> Image.Image:
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image
    width, height = pil_image.size
    crop_width, crop_height = max(1, int(width * crop_ratio)), max(1, int(height * crop_ratio))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), getattr(Image, "Resampling", Image).BILINEAR)


def messages(frames: dict[str, list[np.ndarray]], instruction: str) -> list[dict]:
    """The checkpoint's chat prompt: each camera's frames as one video, then the instruction (no reasoning)."""
    content = []
    for label, camera in CAMERAS:
        content += [{"type": "text", "text": label},
                    {"type": "video", "video": [center_crop(f) for f in frames[camera]]}]
    content.append({"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"})
    return [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text",
                                                                                    "text": "<cot></cot>"}]}]


def unpack(images: dict[str, np.ndarray], state: np.ndarray) -> tuple[dict[str, list[np.ndarray]], np.ndarray]:
    """The robot side's history encoding → frames per camera (oldest first) and the (1, LENGTH, STATE_DIM) state."""
    frames = {camera: [images[f"{camera}@{k}"] for k in range(LENGTH)] for _, camera in CAMERAS}
    rows = np.asarray(state, dtype=np.float32).reshape(LENGTH, AGENT_POS)
    padded = np.zeros((1, LENGTH, STATE_DIM), dtype=np.float32)
    padded[0, :, :14] = np.stack([state14(r) for r in rows])
    return frames, padded
