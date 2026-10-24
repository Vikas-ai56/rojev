"""Revision-benchmark scorecard (spec §11, T8): scores one run's records against the changes injected into it, and
aggregates run scores per arm. Robot-agnostic: the caller loads the records as plain dicts (owner events, ticks, S2
decisions, summary); nothing here reads files, knows the robot or imports an executor.

A run is a `Case`: its arm and the changes injected into it (`Injection`: when, what kind, the expected act class and,
for a swap, the text naming the goal before and after). Each injection is judged inside its window: from the change
to the next injection or the run end. Times are the robot clock.

Definitions. "Applied answer" = the first owner event of kind `command`, `reaffirm` or `continue` inside the window
whose `trigger` is the change's (`instruction` for speech, `deviation` for a deviation) and that a decider made; the
owner's own hold for an unanswered change counts only if no decider's answer follows it in the window (an answer
past the interaction deadline is applied late, noted).
  change time        speech: the owner's first `instruction` event at or after `Injection.t`; deviation: `Injection.t`
                     itself (fault onset, so the monitor's detection time counts as reaction time)
  success            summary `task_success` (the simulator's check of the task the run started with); None if absent
  answer latency     applied answer t − change time (robot clock: one control tick of resolution, so an answer within
                     the tick of the change reads 0)
  execution latency  first tick whose `cmd` is the answer's new command − change time; None for a `continue` answer
                     (no new command) or when the command never reached a tick inside the window
  act                class of the applied answer: `continue` / `reaffirm` → continue; command op switch_to / retry →
                     revise (a resume of the interrupted step is a switch_to: revise); guarded_wait → wait; ask → ask;
                     abort → abort; any other op keeps its own name and never matches. The owner's own hold for an
                     unanswered change (no decider, after an `unanswered` event) is no decision: act `unanswered`,
                     never correct, counted in the arm's unanswered rate (a safe fallback is not a right answer)
  act correct        act == expected; with no applied answer: False, or None when less than `answer_window_s` of the
                     window remained (cannot be judged)
  wrong-goal commitments  `command` / `reaffirm` events inside the window whose operation moves the robot and whose
                     subtask names `old_goal` and not `goal` (case-insensitive containment of the case's own text)
  wasted motion      chunk ticks inside the window under a command whose subtask names `old_goal` and not `goal`
                     (the command running at the change included), as ticks and as seconds (ticks / `control_hz`).
                     Ticks carry commanded actions, not measured poses, and which action dims are arm motion is the
                     robot's knowledge, so a robot-agnostic path length is not computable from these records
  pause              hold ticks in [change, change + `pause_window_s`) clipped to the window, and their share of the
                     ticks there. Aggregated apart by what the change asked for: `pause_when_moving` over changes whose
                     expected act keeps the robot moving (continue, revise: any pause is the freeze T8 looks for),
                     `hold_when_asked` over changes expecting wait, ask or abort (holding is the right answer)
  asks               `command` events with op ask inside the window. Ask precision = asks inside windows whose expected
                     act is ask / all asks of the run (asks outside every window count as misses)
  act ECE            per injection, the first interaction-lane decision inside the window with the change's trigger and
                     a `scores` map (act label → calibrated probability, summing to 1). Labels map to classes as their
                     operations do (continue; revise, resume, and retry in runs recorded before 2026-10-24 →
                     revise; wait; ask; abort); a class's probability is the sum of its labels'. Confidence = top class probability, correct = top class == expected.
                     ECE = Σ_k (n_k / N) |accuracy_k − mean confidence_k| over `bins` equal-width bins [k/B, (k+1)/B)
                     (the last closed), pooled over all runs of an arm; None without scored decisions
The goal metrics need both goal texts and subtask text on every command they judge (an arm whose S1 plans its own
subtasks records none): otherwise None. Every None says why in `notes`.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import fmean

from rojev.protocol import Operation, OpKind
from rojev.system2.jev import ABORT, ASK, CONTINUE, RESUME, REVISE, WAIT

ACTS = (CONTINUE, REVISE, WAIT, ASK, ABORT)
MOVING_ACTS = (CONTINUE, REVISE)
UNANSWERED = "unanswered"
KINDS = ("goal_swap", "object_swap", "stop", "pause", "resume", "ambiguous", "harmless", "deviation")
ANSWER_KINDS = ("command", "reaffirm", "continue")
OP_ACT = {OpKind.SWITCH_TO.value: REVISE, OpKind.RETRY.value: REVISE, OpKind.GUARDED_WAIT.value: WAIT,
          OpKind.ASK.value: ASK, OpKind.ABORT.value: ABORT, OpKind.CONTINUE.value: CONTINUE}
RECORDED_RETRY = "retry"
LABEL_ACT = {CONTINUE: CONTINUE, REVISE: REVISE, RECORDED_RETRY: REVISE, RESUME: REVISE, WAIT: WAIT, ASK: ASK,
             ABORT: ABORT}
INSTRUCTION_MATCH_S = 1e-3
PROBABILITY_SUM_TOL = 1e-3


@dataclass(frozen=True)
class Injection:
    """One change injected into a run at `t`. `goal` / `old_goal`: text naming the goal after / before a swap as it
    appears in subtask text (e.g. "plate" / "microwave"), given together; `phase`: where in the task it was injected
    (pre-grasp, contact, transport, pre-commit), a label carried into the scores."""

    t: float
    kind: str
    expected: str
    goal: str | None = None
    old_goal: str | None = None
    phase: str | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r} not in {KINDS}")
        if self.expected not in ACTS:
            raise ValueError(f"expected act {self.expected!r} not in {ACTS}")
        if (self.goal is None) != (self.old_goal is None):
            raise ValueError("goal and old_goal are given together")

    @property
    def trigger(self) -> str:
        return "deviation" if self.kind == "deviation" else "instruction"


@dataclass(frozen=True)
class Case:
    """A benchmark run: its name, the arm that ran it, and its injections in time order."""

    name: str
    arm: str
    injections: tuple[Injection, ...]

    def __post_init__(self):
        times = [i.t for i in self.injections]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"case {self.name!r}: injections must be in strictly increasing time order")


@dataclass(frozen=True)
class RunRecords:
    """One run's records as loaded by the caller: `summary.json`, `owner.jsonl` (events), `ticks.jsonl`,
    `decisions.jsonl` (both lanes)."""

    summary: dict
    events: list[dict]
    ticks: list[dict]
    decisions: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class InjectionScore:
    injection: Injection
    change_t: float | None
    answer_t: float | None
    answer_kind: str | None
    answer_op: str | None
    answer_lane: str | None
    answer_decider: str | None
    act: str | None
    correct: bool | None
    answer_latency_s: float | None
    execution_latency_s: float | None
    unanswered: int
    wrong_goal_commitments: int | None
    wasted_motion_ticks: int | None
    wasted_motion_s: float | None
    pause_ticks: int | None
    pause_fraction: float | None
    asks: int
    confidence: float | None
    confident_correct: bool | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class RunScore:
    case: str
    arm: str
    success: bool | None
    injections: tuple[InjectionScore, ...]
    asks: int
    asks_expected: int

    def calibration_pairs(self) -> list[tuple[float, bool]]:
        return [(s.confidence, s.confident_correct) for s in self.injections if s.confidence is not None]


@dataclass(frozen=True)
class Stat:
    """An aggregate and how many values it is over (None when there are none)."""

    value: float | None
    n: int


@dataclass(frozen=True)
class ArmScore:
    arm: str
    runs: int
    changes: int
    missed: int
    success: Stat
    answer_latency_s: Stat
    execution_latency_s: Stat
    act_accuracy: Stat
    wrong_goal_commitments: Stat
    wasted_motion_s: Stat
    pause_when_moving: Stat
    hold_when_asked: Stat
    unanswered_rate: Stat
    ask_precision: Stat
    act_ece: Stat


def expected_calibration_error(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    """ECE over (confidence, correct) pairs in `bins` equal-width bins; None for no pairs."""
    if not pairs:
        return None
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for confidence, correct in pairs:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence {confidence} outside [0, 1]")
        buckets[min(int(confidence * bins), bins - 1)].append((confidence, correct))
    return sum(len(b) / len(pairs) * abs(fmean(float(ok) for _, ok in b) - fmean(c for c, _ in b))
               for b in buckets.values())


def score_run(case: Case, records: RunRecords, *, pause_window_s: float = 3.0,
              answer_window_s: float = 2.0) -> RunScore:
    """Scores every injection of `case` on its run's records (definitions in the module docstring)."""
    end = records.summary["sim_s"]
    stops = [i.t for i in case.injections[1:]] + [end]
    scores = tuple(_score_injection(inj, min(stop, end), records, pause_window_s, answer_window_s)
                   for inj, stop in zip(case.injections, stops))
    asks = sum(1 for e in records.events if e["kind"] == "command" and e.get("op") == OpKind.ASK.value)
    expected = sum(s.asks for s in scores if s.injection.expected == ASK)
    success = records.summary.get("task_success")
    return RunScore(case.name, case.arm, None if success is None else bool(success), scores, asks, expected)


def aggregate(runs: list[RunScore], bins: int = 10) -> list[ArmScore]:
    """One row per arm, in first-seen order: means over the values each metric could judge, with their counts."""
    arms: dict[str, list[RunScore]] = defaultdict(list)
    for r in runs:
        arms[r.arm].append(r)
    return [_arm_score(arm, group, bins) for arm, group in arms.items()]


def render_json(arms: list[ArmScore], runs: list[RunScore] = ()) -> dict:
    """JSON-ready dict: the per-arm table and, when given, every run's per-injection scores."""
    return {"arms": [asdict(a) for a in arms], "runs": [asdict(r) for r in runs]}


def render_markdown(arms: list[ArmScore], runs: list[RunScore] = ()) -> str:
    """The per-arm table (each cell `value (n)`) and, when runs are given, one row per injection."""
    head = ("| arm | runs | changes | missed | success | answer s | execution s | act acc | unanswered "
            "| wrong-goal | wasted s | pause when moving | hold when asked | ask precision | act ECE |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for a in arms:
        cells = [a.arm, str(a.runs), str(a.changes), str(a.missed)] + [_cell(s) for s in (
            a.success, a.answer_latency_s, a.execution_latency_s, a.act_accuracy, a.unanswered_rate,
            a.wrong_goal_commitments, a.wasted_motion_s, a.pause_when_moving, a.hold_when_asked, a.ask_precision,
            a.act_ece)]
        lines.append("| " + " | ".join(cells) + " |")
    if runs:
        row_head = ("| case | arm | t | kind | phase | expected | act | answer s | execution s | wrong-goal "
                    "| wasted s | pause | notes |")
        lines += ["", row_head, "|" + "---|" * (row_head.count("|") - 1)]
        for r in runs:
            for s in r.injections:
                i = s.injection
                cells = [r.case, r.arm, f"{i.t:g}", i.kind, i.phase or "", i.expected, s.act or "—",
                         _num(s.answer_latency_s), _num(s.execution_latency_s), _num(s.wrong_goal_commitments),
                         _num(s.wasted_motion_s), _num(s.pause_fraction), "; ".join(s.notes)]
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _score_injection(inj: Injection, stop: float, records: RunRecords, pause_window_s: float,
                     answer_window_s: float) -> InjectionScore:
    change = _change_time(inj, stop, records.events)
    if change is None:
        why = "change at or after the run end" if inj.t >= records.summary["sim_s"] else \
            "no instruction event for this change in the records"
        return InjectionScore(inj, None, None, None, None, None, None, None, None, None, None, 0, None, None, None,
                              None, None, 0, None, None, (why,))
    notes: list[str] = []
    window = [e for e in records.events if change <= e["t"] < stop]
    answers = [e for e in window if e["kind"] in ANSWER_KINDS and e.get("trigger") == inj.trigger]
    decided = next((e for e in answers if e.get("decider") is not None), None)
    answer = decided or (answers[0] if answers else None)
    act = correct = answer_latency = execution_latency = None
    unanswered = sum(1 for e in window if e["kind"] == "unanswered" and e.get("trigger") == inj.trigger)
    if decided is not None and unanswered:
        notes.append("answered late, after the owner's safety hold")
    if answer is not None and unanswered and answer.get("decider") is None:
        act, correct = UNANSWERED, False
        answer_latency = answer["t"] - change
        notes.append("no decider answered: the owner's safety hold")
    elif answer is not None:
        act = CONTINUE if answer["kind"] in ("reaffirm", "continue") else OP_ACT.get(answer["op"], answer["op"])
        correct = act == inj.expected
        answer_latency = answer["t"] - change
        execution_latency = _execution_latency(answer, change, stop, records.ticks, notes)
    elif stop - change < answer_window_s:
        notes.append(f"no answer and only {stop - change:.2f} s of the window left")
    else:
        correct = False
        notes.append("no applied answer")
    wrong, wasted = _goal_metrics(inj, window, change, stop, records, notes)
    hz = records.summary.get("control_hz")
    wasted_s = None if wasted is None or not hz else wasted / hz
    if wasted is not None and not hz:
        notes.append("no control_hz in the summary: wasted motion in ticks only")
    pause_ticks, pause_fraction = _pause(change, min(change + pause_window_s, stop), records.ticks)
    asks = sum(1 for e in window if e["kind"] == "command" and e.get("op") == OpKind.ASK.value)
    confidence, confident_correct = _calibration(inj, change, stop, records.decisions, notes)
    found = answer or {}
    return InjectionScore(inj, change, found.get("t"), found.get("kind"), found.get("op"), found.get("lane"),
                          found.get("decider"), act, correct, answer_latency, execution_latency, unanswered, wrong,
                          wasted, wasted_s, pause_ticks, pause_fraction, asks, confidence, confident_correct,
                          tuple(notes))


def _change_time(inj: Injection, stop: float, events: list[dict]) -> float | None:
    if inj.trigger == "deviation":
        return inj.t if inj.t < stop else None
    return next((e["t"] for e in events if e["kind"] == "instruction"
                 and inj.t - INSTRUCTION_MATCH_S <= e["t"] < stop), None)


def _execution_latency(answer: dict, change: float, stop: float, ticks: list[dict], notes: list[str]) -> float | None:
    if answer["kind"] == "continue":
        return None
    first = next((t["t"] for t in ticks if answer["t"] <= t["t"] < stop and t["cmd"] == answer["cmd"]), None)
    if first is None:
        notes.append(f"cmd {answer['cmd']} reached no tick inside the window")
        return None
    return first - change


def _goal_metrics(inj: Injection, window: list[dict], change: float, stop: float, records: RunRecords,
                  notes: list[str]) -> tuple[int | None, int | None]:
    if inj.goal is None:
        notes.append("no goal text in the case: goal metrics not judged")
        return None, None
    old, new = inj.old_goal.lower(), inj.goal.lower()

    def wrong(subtask: str) -> bool:
        return old in subtask.lower() and new not in subtask.lower()

    moving = [e for e in window if e["kind"] in ("command", "reaffirm") and Operation(OpKind(e["op"])).moves_robot()]
    commitments = None
    if any(e.get("subtask") is None for e in moving):
        notes.append("a moving command without subtask text (S1 plans its own): wrong-goal commitments not judged")
    else:
        commitments = sum(1 for e in moving if wrong(e["subtask"]))
    subtasks = {e["cmd"]: e.get("subtask") for e in records.events if e["kind"] in ("command", "reaffirm")}
    chunk = [t for t in records.ticks if change <= t["t"] < stop and t["source"] == "chunk"]
    blind = sorted({t["cmd"] for t in chunk if subtasks.get(t["cmd"]) is None}, key=str)
    if blind:
        notes.append(f"chunk ticks under commands without subtask text {blind}: wasted motion not judged")
        return commitments, None
    return commitments, sum(1 for t in chunk if wrong(subtasks[t["cmd"]]))


def _pause(start: float, stop: float, ticks: list[dict]) -> tuple[int | None, float | None]:
    inside = [t for t in ticks if start <= t["t"] < stop]
    if not inside:
        return None, None
    held = sum(1 for t in inside if t["source"] == "hold")
    return held, held / len(inside)


def _calibration(inj: Injection, change: float, stop: float, decisions: list[dict],
                 notes: list[str]) -> tuple[float | None, bool | None]:
    scored = next((d for d in decisions if d.get("lane") == "interaction" and d.get("trigger") == inj.trigger
                   and change <= d["t_start"] < stop and d.get("scores")), None)
    if scored is None:
        notes.append("no scored interaction decision: not in act ECE")
        return None, None
    total = sum(scored["scores"].values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOL:
        raise ValueError(f"request {scored.get('request')}: scores sum to {total}, not a probability distribution")
    classes: dict[str, float] = defaultdict(float)
    for label, p in scored["scores"].items():
        if label not in LABEL_ACT:
            raise ValueError(f"request {scored.get('request')}: unknown act label {label!r}")
        classes[LABEL_ACT[label]] += p
    top = max(ACTS, key=lambda a: classes.get(a, 0.0))
    return classes[top], top == inj.expected


def _arm_score(arm: str, group: list[RunScore], bins: int) -> ArmScore:
    changes = [s for r in group for s in r.injections if s.change_t is not None]
    asks = sum(r.asks for r in group)
    pairs = [p for r in group for p in r.calibration_pairs()]
    return ArmScore(
        arm, len(group), len(changes), sum(1 for s in changes if s.act is None and s.correct is False),
        _mean(None if r.success is None else float(r.success) for r in group),
        _mean(s.answer_latency_s for s in changes),
        _mean(s.execution_latency_s for s in changes),
        _mean(None if s.correct is None else float(s.correct) for s in changes),
        _mean(s.wrong_goal_commitments for s in changes),
        _mean(s.wasted_motion_s for s in changes),
        _mean(s.pause_fraction for s in changes if s.injection.expected in MOVING_ACTS),
        _mean(s.pause_fraction for s in changes if s.injection.expected not in MOVING_ACTS),
        _mean(float(s.unanswered > 0) for s in changes),
        Stat(sum(r.asks_expected for r in group) / asks if asks else None, asks),
        Stat(expected_calibration_error(pairs, bins), len(pairs)))


def _mean(values) -> Stat:
    known = [v for v in values if v is not None]
    return Stat(fmean(known) if known else None, len(known))


def _cell(s: Stat) -> str:
    return f"{_num(s.value)} ({s.n})"


def _num(v) -> str:
    if v is None:
        return "—"
    return str(v) if isinstance(v, int) else f"{v:.3f}"
