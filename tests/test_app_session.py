"""The shared assembly with a stand-in robot (a stepped two-dimension backend: one motion, one gripper switch), a
stand-in S1, stand-in deciders and a stand-in done checker: a lockstep run writes every record and keeps time at the
control rate; a step planner with a done checker driven by gripper cues finishes a two-step task and stops; a user
instruction is acknowledged; `stop_when` ends a run; a failing backend ends it as an error with its records written;
plug-ins and settings load from configuration and refuse what they do not know."""
import json

import numpy as np
import pytest

from rojev.app import Embodiment, Session, SessionConfig, build, load_config
from rojev.eval import NativePlanner
from rojev.protocol import Observation
from rojev.robot import KEEP, ActionDim, ActionSpec
from rojev.system1 import ActionChunk, Policy
from rojev.system2 import COMPLETE_STEP, NextStep, StepPlanner

SPEC = ActionSpec("stand-in", (ActionDim("v", -1, 1, 0.0),
                               ActionDim("gripper", -1, 1, KEEP, threshold=0.0, irreversible=True)), control_hz=20)
EMBODIMENT = Embodiment("stand-in", SPEC, ("top",), np.array([0.0, -1.0]))
STEPS = ["pick the cube", "place the cube"]


class SteppedRobot:
    def __init__(self, fail_after=None):
        self.steps, self.fail_after = 0, fail_after

    def observe(self):
        return Observation({"top": np.zeros((8, 8, 3), np.uint8)}, np.zeros(2, np.float32), "put the cube in A")

    def step(self, action):
        self.steps += 1
        if self.fail_after is not None and self.steps > self.fail_after:
            raise RuntimeError("simulator crashed")
        return self.observe()


class GripperPolicy(Policy):
    """Each chunk closes the gripper if the last one opened it, so every chunk boundary is a switch cue."""

    def __init__(self):
        self.calls = 0

    def act(self, obs, subtask, timeout_s=None, continuation=None):
        self.calls += 1
        gripper = 1.0 if self.calls % 2 else -1.0
        return ActionChunk(np.tile([0.1, gripper], (10, 1)).astype(np.float32))

    def reset(self):
        pass


class ScriptedNextStep:
    name = "scripted"

    def next_step(self, goal, record, options, images):
        remaining = [s for s in STEPS if s not in record]
        return NextStep(remaining[0] if remaining else COMPLETE_STEP, f"done so far: {record}")


class AlwaysDone:
    name = "always-done"

    def check(self, images, steps):
        return {s: {"done": 0.95, "possible": 0.05, "blocked": 0.0} for s in steps}


def session(tmp_path, planner=None, backend=None, **kw):
    config = SessionConfig(lockstep=True, watchdog_s=5.0, verify_every_s=30.0, recheck_s=0.05,
                           **kw.pop("config", {}))
    return Session(EMBODIMENT, backend or SteppedRobot(), GripperPolicy(), planner or NativePlanner(),
                   "put the cube in A", config, tmp_path / "run", **kw)


def test_a_lockstep_run_writes_every_record_and_keeps_the_control_rate(tmp_path):
    result = session(tmp_path).run(2.0)
    assert result.ended == "seconds" and abs(result.ticks - 40) <= 1
    names = {p.name for p in (tmp_path / "run").iterdir()}
    assert {"config.json", "owner.jsonl", "ticks.jsonl", "decisions.jsonl", "s1_requests.jsonl", "utterances.jsonl",
            "summary.json", "episodes.jsonl"} <= names
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["starved_ticks"] == 0 and summary["liveness_lapses"] == 0 and summary["error"] is None
    assert [e["outcome"] for e in map(json.loads, (tmp_path / "run" / "episodes.jsonl").read_text().splitlines())] \
        == ["stopped"]


def test_a_step_planner_with_a_cue_driven_done_checker_finishes_a_two_step_task(tmp_path):
    planner = StepPlanner(STEPS, ScriptedNextStep(), cameras=["top"])
    run = session(tmp_path, planner=planner, verifier=AlwaysDone())
    result = run.run(20.0)
    assert result.ended == "complete" and result.complete
    assert result.memory.done() == STEPS and all(r.checked for r in result.memory.records)
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["verify"]["cues"] > 0 and summary["verify"]["events"] == 2
    commands = [c["subtask"] for c in summary["commands"]]
    assert commands[:2] == STEPS


def test_a_user_instruction_is_acknowledged_and_stop_when_ends_the_run(tmp_path):
    robot = SteppedRobot()
    result = session(tmp_path, backend=robot).run(10.0, say=[(0.5, "careful")], stop_when=lambda: robot.steps >= 30)
    assert result.ended == "stop_when" and robot.steps == 30
    said = [json.loads(line)["template"] for line in (tmp_path / "run" / "utterances.jsonl").read_text().splitlines()]
    assert said[0] == "ack"


def test_a_failing_backend_ends_the_run_as_an_error_with_its_records(tmp_path):
    result = session(tmp_path, backend=SteppedRobot(fail_after=5)).run(5.0)
    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert result.ended == "error" and "simulator crashed" in summary["error"]


def test_the_video_has_every_camera_and_the_panel(tmp_path):
    session(tmp_path, video=True).run(1.0)
    assert (tmp_path / "run" / "view.mp4").stat().st_size > 0


def test_plug_ins_and_settings_come_from_configuration_and_refuse_unknown_keys(tmp_path):
    planner = build({"factory": "rojev.eval:NativePlanner"})
    assert planner.name == "native"
    with pytest.raises(ValueError, match="unknown keys"):
        build({"factory": "rojev.eval:NativePlanner", "argz": {}})
    with pytest.raises(ValueError, match="unknown runtime settings"):
        SessionConfig.from_mapping({"lease": 3.0})
    path = tmp_path / "run.yaml"
    path.write_text("runtime:\n  watchdog_s: 4.0\n  lockstep: true\nrobot:\n  factory: x:y\n")
    config, document = load_config(path)
    assert config.watchdog_s == 4.0 and document["robot"]["factory"] == "x:y"
