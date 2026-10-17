"""G0 install gate: simulator, SO-101 asset, offscreen render, and the OneJev GGUF client path.

Run from the repo root: `uv run pytest tests/test_install.py -v`.
The live OneJev test runs only when a `qev serve` instance answers at $QEV_BASE_URL (default http://localhost:8000).
"""
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SO101 = ROOT / "vendor/mujoco_menagerie/robotstudio_so101"
SCENE_BOX = SO101 / "scene_box.xml"
QEV_URL = os.environ.get("QEV_BASE_URL", "http://localhost:8000")

# Recorded from the compiled model, in actuator-index order; each actuator drives the joint of the same name.
EXPECTED_ACTUATORS = [
    ("shoulder_pan", (-1.91986, 1.91986)),
    ("shoulder_lift", (-1.74533, 1.74533)),
    ("elbow_flex", (-1.69, 1.69)),
    ("wrist_flex", (-1.65806, 1.65806)),
    ("wrist_roll", (-2.74385, 2.84121)),
    ("gripper", (-0.17453, 1.74533)),
]


@pytest.fixture(scope="module")
def model():
    import mujoco

    return mujoco.MjModel.from_xml_path(str(SCENE_BOX))


def test_mujoco_version_in_plan_range():
    import mujoco

    major, minor, patch = (int(x) for x in mujoco.__version__.split(".")[:3])
    assert (major, minor, patch) >= (3, 1, 3) and major < 4


def test_mujoco_py_not_installed():
    import importlib.util

    assert importlib.util.find_spec("mujoco_py") is None


def test_so101_assets_are_real_meshes():
    stls = sorted((SO101 / "assets").glob("*.stl"))
    assert len(stls) == 19
    for stl in stls:
        assert not stl.read_bytes()[:40].startswith(b"version https://git-lfs"), stl.name


def test_scene_box_compiles_with_expected_dimensions(model):
    assert (model.nq, model.nv, model.nu) == (13, 12, 6)
    assert model.opt.timestep == pytest.approx(0.005)


def test_actuator_inventory_matches_record(model):
    import mujoco

    for i, (name, (lo, hi)) in enumerate(EXPECTED_ACTUATORS):
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == name
        joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
        assert joint == name
        assert model.actuator_ctrlrange[i].tolist() == pytest.approx([lo, hi])


def test_physics_stable_from_pickup_keyframe(model):
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(400):  # 2 s of simulated time
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert all(w.number == 0 for w in data.warning)


def test_offscreen_render_shows_arm_and_box(model):
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data)
    img = renderer.render().astype(int)
    renderer.close()
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    green_box = (g > 150) & (r < 90) & (b < 90)
    yellow_arm = (r > 150) & (g > 120) & (b < 80)
    assert img.shape == (480, 640, 3)
    assert green_box.sum() > 500, "green box not visible in the default view"
    assert yellow_arm.sum() > 2000, "SO-101 arm not visible in the default view"


def test_llama_server_installed():
    exe = shutil.which("llama-server")
    assert exe is not None
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
    assert "version:" in out.stdout + out.stderr


def test_qev_client_imports_and_cli():
    from qev import Choice, Client  # noqa: F401
    from qev.llamacpp import tokenizer_repo
    from qev.media import data_uri  # noqa: F401

    assert tokenizer_repo("mradermacher/OneJev-0.8B-GGUF:Q8_0") == "OmniJev/OneJev-0.8B"
    assert tokenizer_repo("mradermacher/OneJev1.5-0.8B-GGUF:Q8_0") == "OmniJev/OneJev1.5-0.8B"
    qev = shutil.which("qev", path=str(ROOT / ".venv/bin"))
    assert qev is not None
    assert subprocess.run([qev, "--help"], capture_output=True, timeout=60).returncode == 0


def _qev_up() -> bool:
    try:
        with urllib.request.urlopen(QEV_URL + "/v1/models", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


@pytest.mark.skipif(not _qev_up(), reason=f"no qev server at {QEV_URL}")
def test_onejev_returns_probability_for_every_option(tmp_path):
    import mujoco
    from PIL import Image
    from qev import Choice, Client
    from qev.media import data_uri

    model = mujoco.MjModel.from_xml_path(str(SCENE_BOX))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data)
    png = tmp_path / "observation.png"
    Image.fromarray(renderer.render()).save(png)
    renderer.close()

    options = {
        "approach_box": "Move the gripper towards the visible box",
        "observe": "Observe again because evidence is inadequate",
        "stop": "Do not move",
    }
    r = Client(QEV_URL).system_one(
        state={"instruction": "Approach the visible green box", "screen": "<image:1>",
               "gripper": "open", "previous_skill_result": "none"},
        media=[{"type": "image", "data": data_uri(str(png))}],
        questions={"next": Choice("Which action is most appropriate for the visible scene?", options)},
    )
    probs = r.answers["next"].probabilities
    assert set(probs) == set(options)
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
