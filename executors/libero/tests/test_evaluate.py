"""Evaluation bookkeeping: one session per task episode with the right wiring, a step budget of exactly LeRobot's max
steps, early stop on success, per-task and suite success, results.json; then the whole CLI with the zero policy through
the real `rojev.app.Session` over the fake simulator."""
import json
from pathlib import Path

import numpy as np
import pytest

import fake_libero
from backend import LiberoBackend
from embodiment import EMBODIMENT, START
from evaluate import EpisodeResult, HoldingPolicy, episode_dir, main, render_table, run_episode, summarize
from rojev.app import RunResult, SessionConfig
from rojev.eval import NativePlanner
from rojev.protocol import TaskMemory


class RecordingPolicy(HoldingPolicy):
    def __init__(self):
        super().__init__(START, 20)
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class S1Stats:
    def __init__(self, accepted=1, errors=0):
        self.accepted, self.errors = accepted, errors


class Runner:
    def __init__(self, stats):
        self.stats = stats


class SteppingSession:
    built: list["SteppingSession"] = []
    s1 = S1Stats

    def __init__(self, embodiment, backend, policy, planner, instruction, config, out, video=False):
        self.embodiment, self.backend, self.policy, self.planner = embodiment, backend, policy, planner
        self.instruction, self.config, self.out, self.video = instruction, config, out, video
        self.simulator = backend.env._env
        self.runner = Runner(self.s1())
        SteppingSession.built.append(self)

    def run(self, seconds, say=(), stop_when=None) -> RunResult:
        self.seconds, ticks, ended = seconds, 0, "seconds"
        while ticks * self.embodiment.spec.dt < seconds:
            if stop_when is not None and stop_when():
                ended = "stop_when"
                break
            self.backend.step(START)
            ticks += 1
        sim_s = ticks * self.embodiment.spec.dt
        return RunResult(ended, sim_s, 0.01, ticks, False, TaskMemory(self.instruction), self.out)


def succeeding_at(step: int):
    def open_backend(suite, task, episode, seed):
        backend = LiberoBackend(suite, task, episode, seed)
        backend.env._env.success_at = {backend.env.num_steps_wait + step}
        return backend
    return open_backend


def test_an_episode_is_one_session_wired_for_lerobots_rollout(tmp_path):
    policy = RecordingPolicy()
    config = SessionConfig()
    result = run_episode("libero_goal", 1, 3, 1003, policy, config, tmp_path / "e", video=True,
                         open_backend=succeeding_at(5), session_type=SteppingSession)
    session = SteppingSession.built[-1]
    assert isinstance(session.planner, NativePlanner) and session.policy is policy and session.config is config
    assert session.embodiment is EMBODIMENT and session.video is True and session.out == tmp_path / "e"
    assert session.instruction == "put the bowl on the stove"
    assert session.seconds == 300 * EMBODIMENT.spec.dt
    assert ("seed", 1003) in session.simulator.calls and ("set_init_state", 3) in session.simulator.calls
    assert session.simulator.calls[-1] == ("close",)
    assert policy.resets == 1
    assert result == EpisodeResult(1, 3, 1003, "put the bowl on the stove", True, 5, 5, 300, result.sim_s, 0.01,
                                   "stop_when", str(tmp_path / "e"), s1_accepted=1)


def test_an_episode_without_success_runs_exactly_the_step_budget_and_closes_the_simulator(tmp_path):
    backends: list[LiberoBackend] = []

    def open_backend(*args):
        backends.append(LiberoBackend(*args))
        return backends[-1]

    result = run_episode("libero_spatial", 0, 0, 1000, RecordingPolicy(), SessionConfig(), tmp_path, False,
                         open_backend=open_backend, session_type=SteppingSession)
    assert not result.success and result.success_step is None and result.ended == "seconds"
    assert result.steps == result.max_steps == 280
    assert backends[0].env._env is None


def row(task: int, episode: int, success: bool) -> EpisodeResult:
    return EpisodeResult(task, episode, 1000 + episode, f"task {task}", success, 4 if success else None, 300, 300,
                         15.0, 2.0, "stop_when" if success else "seconds", "out")


def test_success_is_summarized_per_task_and_for_the_suite():
    summary = summarize([row(0, 0, True), row(0, 1, False), row(1, 0, True), row(1, 1, True)])
    assert [(t["task"], t["successes"], t["episodes"], t["success_rate"]) for t in summary["tasks"]] == \
        [(0, 1, 2, 0.5), (1, 2, 2, 1.0)]
    assert (summary["successes"], summary["episodes"], summary["success_rate"]) == (3, 4, 0.75)
    table = render_table("libero_goal", summary)
    assert "libero_goal: 3/4 = 0.7500" in table and "task 1" in table


def test_the_cli_scores_a_zero_policy_through_the_real_session(tmp_path, monkeypatch):
    original_init = fake_libero.OffScreenRenderEnv.__init__

    def init_with_success(self, **kwargs):
        original_init(self, **kwargs)
        self.success_at = {10 + 4} if "middle_drawer" in kwargs["bddl_file_name"] else set()

    monkeypatch.setattr(fake_libero.OffScreenRenderEnv, "__init__", init_with_success)
    out = tmp_path / "run"
    assert main(["--policy", "zero", "--suite", "libero_goal", "--tasks", "0", "2", "--episodes", "1",
                 "--out", str(out)]) == 0
    results = json.loads((out / "results.json").read_text())
    assert results["suite"] == "libero_goal" and results["runtime"]["lockstep"] is True
    assert [(r["task"], r["success"], r["steps"], r["ended"]) for r in results["runs"]] == \
        [(0, True, 4, "stop_when"), (2, False, 300, "seconds")]
    assert results["success_rate"] == 0.5
    session = episode_dir(out, 0, 0)
    for name in ("config.json", "summary.json", "ticks.jsonl", "s1_requests.jsonl", "decisions.jsonl"):
        assert (session / name).is_file(), name
    ticks = [json.loads(line) for line in (episode_dir(out, 2, 0) / "ticks.jsonl").read_text().splitlines()]
    assert len(ticks) == 300
    assert all(np.allclose(t["action"], START) for t in ticks)


def test_the_shipped_real_time_config_runs_an_episode_with_the_simulator_on_its_own_thread(tmp_path, monkeypatch):
    """Lockstep off (config/runs/libero-realtime.yaml): the control loop steps the simulator from its own thread at
    wall-clock pace, the backend hands every call to its simulator thread, and success still ends the episode."""
    original_init = fake_libero.OffScreenRenderEnv.__init__

    def init_with_success(self, **kwargs):
        original_init(self, **kwargs)
        self.success_at = {10 + 4}

    monkeypatch.setattr(fake_libero.OffScreenRenderEnv, "__init__", init_with_success)
    out = tmp_path / "run"
    config = Path(__file__).resolve().parents[3] / "config/runs/libero-realtime.yaml"
    assert main(["--policy", "zero", "--suite", "libero_goal", "--tasks", "0", "--episodes", "1", "--config",
                 str(config), "--out", str(out)]) == 0
    results = json.loads((out / "results.json").read_text())
    assert results["runtime"]["lockstep"] is False
    assert [(r["success"], r["success_step"], r["ended"]) for r in results["runs"]] == [(True, 4, "stop_when")]
    assert 0.15 <= results["runs"][0]["wall_s"] < 5.0


@pytest.fixture(autouse=True)
def forget_sessions():
    SteppingSession.built.clear()



class BrokenS1Session(SteppingSession):
    s1 = staticmethod(lambda: S1Stats(accepted=0, errors=5))


def test_an_episode_where_s1_never_answers_is_stopped_and_flagged_broken(tmp_path):
    result = run_episode("libero_goal", 0, 0, 1000, RecordingPolicy(), SessionConfig(), tmp_path, False,
                         open_backend=succeeding_at(1000), session_type=BrokenS1Session)
    assert result.s1_broken and result.ended == "stop_when" and result.steps == 0
