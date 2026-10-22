"""rojev.sim: scene composition, contacts and segmentation ground truth."""
import mujoco
import pytest

from rojev.robot.sim import world as sim


@pytest.fixture()
def scene():
    m = sim.compose("pick_place.xml")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_compose_keeps_robot_solver_options_and_adds_world(scene):
    m, _ = scene
    robot = mujoco.MjModel.from_xml_path(str(sim.SO101_DIR / "so101.xml"))
    for key, field in (("timestep", "timestep"), ("iterations", "iterations"), ("ls_iterations", "ls_iterations"),
                       ("impratio", "impratio"), ("integrator", "integrator"), ("cone", "cone")):
        assert getattr(m.opt, field) == getattr(robot.opt, field), key
    assert m.nu == 6 and {m.camera(i).name for i in range(m.ncam)} == {"overhead", "wrist_cam"}
    assert m.body("box") is not None


def test_compose_requires_the_mount_frame(tmp_path):
    world = tmp_path / "w.xml"
    world.write_text("<mujoco><worldbody><geom type='plane' size='1 1 .1'/></worldbody></mujoco>")
    with pytest.raises(ValueError, match="robot_mount"):
        sim.compose(world)


def test_contacts_with_floor_counts_only_non_ignored_bodies(scene):
    m, d = scene
    for _ in range(100):  # let the box settle onto the floor
        mujoco.mj_step(m, d)
    assert sim.contacts_with(m, d, "floor", ignore_bodies=("box",)) == 0
    assert sim.contacts_with(m, d, "floor") > 0


def test_visible_pixels_is_zero_for_invisible_geom_and_restores_rgb(scene):
    m, d = scene
    r = mujoco.Renderer(m, height=240, width=320)
    shown = sim.visible_pixels(m, r, d, "box", camera="overhead")
    m.geom_rgba[m.geom("box").id, 3] = 0.0
    hidden = sim.visible_pixels(m, r, d, "box", camera="overhead")
    r.update_scene(d, camera="overhead")
    frame = r.render()
    r.close()
    assert shown > 100 and hidden == 0
    assert frame.max() > 1  # back to RGB, not segmentation ids
