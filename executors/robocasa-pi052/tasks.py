"""Workflow task set (RoboCasa pretrain split): authored subtask sequences + ground-truth completion checks.

Used by arm B (fixed sequence, fixed step budget) and arm E (same sequence, switch when the sim says the subtask is
done), and logged for every arm as the reference: per step we record which subtasks are truly done.
- Subtask wording = the most frequent phrasing in π0.5²'s training subtasks (counts in comments, E8 vocabulary);
  object names come from the scene (`get_obj_lang`).
- Completion checks reuse RoboCasa's own success primitives (the parts of each task's `_check_success`).
- `chunks_per_subtask` (arm B) = p75 subtask duration in the task's training demos ÷ 50-step chunks, rounded up;
  `max_steps` ≈ 1.5 × p90 demo length (docs/features/v2-levers.md §7).
"""
from collections.abc import Callable
from dataclasses import dataclass

from robocasa.utils import object_utils as OU


@dataclass(frozen=True)
class Step:
    subtask: str
    done: Callable[[object], bool]  # kitchen env -> ground truth


@dataclass(frozen=True)
class TaskSpec:
    env_task: str
    max_steps: int
    chunks_per_subtask: int
    steps: Callable[[object], list[Step]]  # built after reset: object names and scene-dependent order


def _placed(k, obj: str, inside: Callable[[object], bool]) -> bool:
    return inside(k) and OU.gripper_obj_far(k, obj)


def _picked(k, obj: str, source) -> bool:
    """Grasped AND lifted: no contact with the fixture it started in or with its container (plate), if any.

    Grasp alone is not "picked up": closing the gripper on an object still resting on its plate fires
    `check_obj_grasped` (first E run, 2026-10-18: switched to "place" at step ~300, steak never left the plate).
    """
    body = k.objects[obj]
    container = k.objects.get(f"{obj}_container")  # RoboCasa names the plate a `try_to_place_in` object sits on
    return (OU.check_obj_grasped(k, obj) and not k.check_contact(body, source)
            and (container is None or not k.check_contact(body, container)))


def microwave_thawing_fridge(k) -> list[Step]:
    """Fridge and microwave doors start open. Success: meat inside, gripper away, microwave on."""
    meat = k.get_obj_lang("meat")
    return [
        Step(f"pick up the {meat}", lambda k: _picked(k, "meat", k.fridge)),  # "pick up the steak" 448
        Step(f"place the {meat} in the microwave",  # 24
             lambda k: _placed(k, "meat", lambda k: OU.obj_inside_of(k, "meat", k.microwave))),
        Step("close the microwave door", lambda k: k.microwave.is_closed(k)),  # 1113
        Step("press the start button", lambda k: k.microwave.get_state()["turned_on"]),  # 435 (vs "turn on the microwave" 27)
    ]


def move_fridge_to_freezer(k) -> list[Step]:
    """Fridge and freezer doors start open. Success: item on a freezer rack, both doors closed."""
    item = k.get_obj_lang("refrigerated_item")
    return [
        Step(f"pick up the {item}", lambda k: _picked(k, "refrigerated_item", k.fridge)),  # "pick up the chicken drumstick" 657
        Step(f"place the {item} in the freezer",  # 18
             lambda k: _placed(k, "refrigerated_item",
                               lambda k: k.fridge.check_rack_contact(k, "refrigerated_item", compartment="freezer"))),
        Step("close the freezer door", lambda k: k.fridge.is_closed(k, compartment="freezer")),  # 427
        Step("close the refrigerator door", lambda k: k.fridge.is_closed(k, compartment="fridge")),  # 1430
    ]


def load_condiments_in_fridge(k) -> list[Step]:
    """Fridge starts open. Distractors already on the top shelf are moved down first; then both condiments go up."""
    top = k.target_rack_index
    on_top = lambda k, o: k.fridge.check_rack_contact(k, o, compartment="fridge", rack_index=top)  # noqa: E731
    steps = []
    for d in ("distractor1", "distractor2"):
        if on_top(k, d):  # scene-dependent: decided once, at reset
            name = k.get_obj_lang(d)
            steps += [Step(f"pick up the {name}", lambda k, d=d: _picked(k, d, k.fridge)),
                      Step(f"place the {name} on the middle shelf",  # wording not in the vocabulary (lever)
                           lambda k, d=d: not on_top(k, d) and k.fridge.check_rack_contact(k, d, compartment="fridge")
                           and OU.gripper_obj_far(k, d))]
    for c in ("condiment1", "condiment2"):
        name = k.get_obj_lang(c)
        steps += [Step(f"pick up the {name}", lambda k, c=c: _picked(k, c, k.counter)),  # "pick up the ketchup bottle" 116
                  Step(f"place the {name} on the top shelf",  # "place the ketchup bottle on the top shelf" 25
                       lambda k, c=c: on_top(k, c) and OU.gripper_obj_far(k, c))]
    return steps


TASKS = {t.env_task: t for t in (
    TaskSpec("MicrowaveThawingFridge", max_steps=2320, chunks_per_subtask=5, steps=microwave_thawing_fridge),
    TaskSpec("MoveFridgeToFreezer", max_steps=1520, chunks_per_subtask=5, steps=move_fridge_to_freezer),
    TaskSpec("LoadCondimentsInFridge", max_steps=2370, chunks_per_subtask=4, steps=load_condiments_in_fridge),
)}
