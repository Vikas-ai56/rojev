"""One contract every robot backend must satisfy (simulator and mock now; the real SO-101 later)."""
import mujoco
import numpy as np
import pytest

from rojev import sim
from rojev.robot import CameraSpec, JointLimits, MockRobot, MujocoRobot, Rate, camera_info, so101


def _mujoco():
    return so101.simulated(cameras=(CameraSpec("overhead", width=320, height=240, fps=10.0, depth=True, latency_s=0.05),))


def _mock():
    model = sim.compose("pick_place.xml")
    names = tuple(model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu))
    return MockRobot(JointLimits.from_model(model, names, {n: 4.4 for n in names}, "test"))


@pytest.fixture(params=[_mujoco, _mock], ids=["mujoco", "mock"])
def robot(request):
    with request.param() as r:
        yield r


def _ramp_to(robot, target, seconds):
    """Command a straight-line ramp at the control rate, staying inside the velocity limit."""
    rate, start = Rate(robot.clock, robot.control_rate_hz), robot.read().joints.position
    n = int(seconds * robot.control_rate_hz)
    results = []
    for k in range(1, n + 1):
        results.append(robot.write(robot.command(start + (target - start) * k / n)))
        rate.sleep()
        robot.read()
    return results


def test_names_and_limits_are_consistent(robot):
    assert robot.joint_names == robot.limits.names
    assert np.all(robot.limits.lower < robot.limits.upper)
    assert len(robot.joint_names) == 6


def test_write_before_read_is_rejected(robot):
    with pytest.raises(RuntimeError, match="before any read"):
        robot.write(robot.command(np.zeros(6)))


def test_stamps_follow_the_robot_clock(robot):
    rate = Rate(robot.clock, robot.control_rate_hz)
    obs = [robot.read()]
    for _ in range(5):
        rate.sleep()
        obs.append(robot.read())
    seqs = [o.joints.stamp.seq for o in obs]
    times = [o.joints.stamp.acquired for o in obs]
    assert seqs == list(range(seqs[0], seqs[0] + 6))
    assert np.allclose(np.diff(times), 1 / robot.control_rate_hz, atol=0.006)  # sim time moves in 5 ms physics steps
    assert times[-1] == pytest.approx(robot.clock.now())


def test_in_limit_ramp_has_no_violations_and_converges(robot):
    target = (robot.limits.lower + robot.limits.upper) / 2 + 0.2
    results = _ramp_to(robot, target, seconds=1.0)
    assert all(not r.violations for r in results)
    for _ in range(int(0.5 * robot.control_rate_hz)):  # settle
        robot.write(robot.command(target))
        Rate(robot.clock, robot.control_rate_hz).sleep()
        robot.read()
    assert np.abs(robot.read().joints.position - target).max() < 0.01


def test_out_of_range_target_is_clamped_and_reported(robot):
    robot.read()
    q = robot.read().joints.position.copy()
    bad = q.copy()
    bad[0] = robot.limits.upper[0] + 1.0
    r = robot.write(robot.command(bad))
    assert any("shoulder_pan" in v and "outside" in v for v in r.violations)
    assert r.applied.position[0] <= robot.limits.upper[0]


def test_large_step_is_limited_to_max_velocity_and_reported(robot):
    q = robot.read().joints.position.copy()
    jump = q.copy()
    jump[1] = np.clip(q[1] + 1.0, robot.limits.lower[1], robot.limits.upper[1])
    r = robot.write(robot.command(jump))
    step = robot.limits.max_velocity[1] / robot.control_rate_hz
    assert any("shoulder_lift step" in v for v in r.violations)
    assert abs(r.applied.position[1] - q[1]) == pytest.approx(step)


def test_camera_frames_respect_rate_latency_and_calibration():
    with _mujoco() as robot:
        rate, frames = Rate(robot.clock, robot.control_rate_hz), []
        for _ in range(int(robot.control_rate_hz)):  # 1 s
            f = robot.read().cameras.get("overhead")
            if f is not None and (not frames or f.stamp.seq != frames[-1].stamp.seq):
                frames.append(f)
            rate.sleep()
        assert 9 <= len(frames) <= 10  # 10 fps camera; the first frame arrives after the 50 ms latency
        assert all(f.stamp.received - f.stamp.acquired == pytest.approx(0.05, abs=robot.model.opt.timestep) for f in frames)
        f = frames[-1]
        assert f.rgb.shape == (240, 320, 3) and f.depth.shape == (240, 320) and f.depth.dtype == np.float32
        assert f.info.parent_frame == "world"
        R = f.info.T_parent_optical[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) == pytest.approx(1.0)


def test_camera_calibration_projects_the_box_onto_its_pixels():
    """K and the extrinsics must map the box's true position onto where the renderer draws it."""
    # zero latency, so the frame and the ground truth are from the same instant
    with so101.simulated(cameras=(CameraSpec("overhead", width=320, height=240, depth=True),)) as robot:
        f = robot.read().cameras["overhead"]
        gt = robot.ground_truth
        T_world_optical = f.info.T_parent_optical  # parent is the world for this camera
        p_world = gt.body_pose("box")[:3, 3] + np.array([0, 0, 0.03])  # top face centre, the part the camera sees
        p_cam = np.linalg.inv(T_world_optical) @ np.append(p_world, 1.0)
        u, v, w = f.info.K @ p_cam[:3]
        r = mujoco.Renderer(robot.model, height=240, width=320)
        r.enable_segmentation_rendering()
        r.update_scene(robot.data, camera="overhead")
        seg = r.render()
        r.close()
        ys, xs = np.nonzero((seg[..., 0] == robot.model.geom("box").id) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
        assert abs(u / w - (xs.mean() + 0.5)) < 1.0 and abs(v / w - (ys.mean() + 0.5)) < 1.0
        # depth at the box centre equals the optical-frame z of its top face
        assert f.depth[int(v / w), int(u / w)] == pytest.approx(p_cam[2], abs=0.003)


def test_camera_info_rejects_unsupported_cameras():
    m = sim.compose("pick_place.xml")
    with pytest.raises(ValueError, match="not fixed"):
        cam = m.camera("overhead").id
        m.cam_mode[cam] = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
        camera_info(m, cam, 64, 48)


def test_backend_rejects_non_position_actuators():
    m = sim.compose("pick_place.xml")
    m.actuator_biastype[0] = mujoco.mjtBias.mjBIAS_NONE
    with pytest.raises(ValueError, match="not a position servo"):
        MujocoRobot(m)


def test_trajectory_violations_flag_position_and_speed():
    lim = _mock().limits
    t = np.array([0.0, 0.1, 0.2])
    ok = np.zeros((3, 6))
    assert lim.trajectory_violations(t, ok) == []
    fast = ok.copy()
    fast[1, 2] = 0.5  # 0.5 rad in 0.1 s = 5 rad/s > 4.4
    out = lim.trajectory_violations(t, fast)
    assert any("elbow_flex speed 5.000" in v for v in out)
    beyond = ok.copy()
    beyond[2, 0] = lim.upper[0] + 0.1
    assert any(v.startswith("t=0.200s shoulder_pan=") and "outside" in v for v in lim.trajectory_violations(t, beyond))
