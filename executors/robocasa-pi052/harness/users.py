"""Simulated users for campaigns: what a person says during a run, and what each line means.

A `Line` is one utterance at a simulated time with its `intent` (what the person wants) and, for a new task, the
trained sentence the robot should choose (`task`). What counts as a right answer depends on the robot's state when
the line is heard (a "go on" resumes a hold but is a plain continue while moving), so `judge.allowed` decides it
from the intent and the recorded state, not here.

`say(intent, rng, task)` draws a phrasing for an intent and roughens it the way people type: typos, missing
punctuation, fillers, casing (`noisy`). `FuzzUser` draws whole sessions: how many lines, which intents, when; its
seed is recorded so a session can be replayed, but each campaign draws new ones (non-deterministic across runs).
"""
import random
from dataclasses import asdict, dataclass

REMARK, MODIFIER, PAUSE, RESUME, STOP, VAGUE = "remark", "modifier", "pause", "resume", "stop", "vague"
UNLISTED, GIBBERISH, STATUS, NEW_TASK, NEGATION, THANKS = ("unlisted", "gibberish", "status", "new_task", "negation",
                                                           "thanks")

PHRASES = {
    REMARK: ["nice", "looking good", "good job, keep going", "ok", "nice work robot", "that's the way", "cool"],
    MODIFIER: ["a bit slower please", "be careful", "go faster", "make it fast pls", "gently", "take your time"],
    PAUSE: ["wait", "hold on", "pause for a sec", "wait a moment", "hang on", "one second, stop moving"],
    RESUME: ["go on", "ok continue", "carry on", "you can continue now", "keep going now", "alright, resume"],
    STOP: ["stop", "stop now", "stop everything", "cancel that", "abort the task", "no stop!"],
    VAGUE: ["put it there", "no, the other one", "over there", "that one", "move it a bit"],
    UNLISTED: ["make me a sandwich", "do a little dance", "fold the laundry", "go to the living room",
               "water the plants"],
    GIBBERISH: ["asdf qwer", "hmm", "uh", "blorp", "...", "k"],
    STATUS: ["are you done?", "how long will this take?", "what are you doing?", "is it finished"],
    THANKS: ["thanks, that's all", "great, done", "perfect, thank you"],
}
NEW_TASK_PHRASES = {
    "Turn on the sink faucet.": ["turn on the sink faucet", "actually, just turn on the tap",
                                 "forget that, switch on the faucet", "can you open the water in the sink instead"],
    "Turn off the sink faucet.": ["turn off the sink faucet", "turn the tap off", "shut off the water"],
    "Open the microwave door.": ["open the microwave door", "actually open the microwave",
                                 "never mind that, open the microwave door instead"],
    "Close the microwave door.": ["close the microwave door", "shut the microwave", "close the microwave please"],
    "Open the left drawer.": ["open the left drawer", "pull out the left drawer", "actually, open the drawer on the left"],
    "Close the left drawer.": ["close the left drawer", "push the left drawer shut"],
}
NEGATIONS = {"Turn on the sink faucet.": ["don't turn on the faucet", "do not touch the tap"],
             "Open the microwave door.": ["don't open the microwave", "leave the microwave alone"]}
FILLERS = ["um, ", "uh ", "ok so ", "hey robot, ", ""]
WEIGHTS = {REMARK: 3, MODIFIER: 2, PAUSE: 3, RESUME: 3, STOP: 1, VAGUE: 2, UNLISTED: 2, GIBBERISH: 1, STATUS: 2,
           NEW_TASK: 3, NEGATION: 1}


@dataclass(frozen=True)
class Line:
    t: float
    text: str
    intent: str
    task: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def noisy(text: str, rng: random.Random, typo_rate: float = 0.04) -> str:
    """The text as typed in a hurry: a filler in front, random adjacent swaps / drops, casing, no final period."""
    chars = list(rng.choice(FILLERS) + text)
    i = 0
    while i < len(chars) - 1:
        if chars[i].isalpha() and rng.random() < typo_rate:
            if rng.random() < 0.5:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            else:
                del chars[i]
        i += 1
    out = "".join(chars).rstrip(".")
    return out.upper() if rng.random() < 0.05 else out.lower() if rng.random() < 0.5 else out


def say(intent: str, rng: random.Random, task: str | None = None, rough: bool = True) -> str:
    if intent == NEW_TASK:
        text = rng.choice(NEW_TASK_PHRASES[task])
    elif intent == NEGATION:
        text = rng.choice(NEGATIONS[task])
    else:
        text = rng.choice(PHRASES[intent])
    return noisy(text, rng) if rough else text


class FuzzUser:
    """Draws a session: 2–`max_lines` lines in [`start_s`, `end_s`], at least `gap_s` apart (a burst of three within
    a second now and then), intents by `WEIGHTS`, new tasks and negations about `tasks` (the listed sentences)."""

    def __init__(self, seed: int, tasks: list[str], max_lines: int = 5, gap_s: float = 3.0):
        self.seed, self.rng, self.tasks = seed, random.Random(seed), [t for t in tasks if t in NEW_TASK_PHRASES]
        self.max_lines, self.gap_s = max_lines, gap_s

    def line(self, t: float) -> Line:
        intents = [i for i in WEIGHTS if self.tasks or i not in (NEW_TASK, NEGATION)]
        intent = self.rng.choices(intents, [WEIGHTS[i] for i in intents])[0]
        pool = [t for t in self.tasks if intent != NEGATION or t in NEGATIONS]
        task = self.rng.choice(pool) if intent in (NEW_TASK, NEGATION) and pool else None
        if intent == NEGATION and task is None:
            intent = REMARK
        return Line(round(t, 2), say(intent, self.rng, task), intent, task if intent == NEW_TASK else None)

    def session(self, start_s: float, end_s: float) -> list[Line]:
        n = self.rng.randint(2, self.max_lines)
        times = sorted(self.rng.uniform(start_s, end_s) for _ in range(n))
        spaced = []
        for t in times:
            if not spaced or t - spaced[-1] >= self.gap_s:
                spaced.append(t)
        lines = [self.line(t) for t in spaced]
        if self.rng.random() < 0.2 and spaced:
            t0 = self.rng.uniform(start_s, end_s)
            lines += [self.line(t0 + 0.3 * k) for k in range(3)]
        return sorted(lines, key=lambda line: line.t)
