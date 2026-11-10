"""LIBERO's benchmark suites and their tasks, as LIBERO's own benchmark API lists them, with LeRobot's step budget.

`load_suite(name)` builds a suite from LIBERO's registry (`libero.libero.benchmark.get_benchmark_dict()`, task order 0,
as LeRobot builds it in `lerobot/envs/libero.py:38-46`). `describe(name)` lists its tasks (index, name, language and
number of init states, read with LeRobot's `get_task_init_states`, `envs/libero.py:67-74`, which loads LIBERO's torch
pickles with `weights_only=False`) and the episode budget LeRobot gives the suite (`TASK_SUITE_MAX_STEPS`,
`envs/libero.py:98-104`; None when LeRobot has none, and its env then falls back to its own default).

  uv run python suites.py [SUITE ...]      every registered suite (or the named ones) with its tasks
"""
import argparse
from dataclasses import dataclass

from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, get_task_init_states
from libero.libero import benchmark


@dataclass(frozen=True)
class TaskInfo:
    index: int
    name: str
    language: str
    init_states: int


@dataclass(frozen=True)
class SuiteInfo:
    name: str
    max_steps: int | None
    tasks: tuple[TaskInfo, ...]


def suite_names() -> list[str]:
    return sorted(benchmark.get_benchmark_dict())


def load_suite(name: str):
    registry = benchmark.get_benchmark_dict()
    if name not in registry:
        raise ValueError(f"unknown LIBERO suite {name!r}; registered: {sorted(registry)}")
    return registry[name]()


def describe(name: str) -> SuiteInfo:
    suite = load_suite(name)
    tasks = tuple(TaskInfo(index, task.name, task.language, len(get_task_init_states(suite, index)))
                  for index, task in enumerate(suite.tasks))
    return SuiteInfo(name, TASK_SUITE_MAX_STEPS.get(name), tasks)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suites", nargs="*", help="suite names (default: every registered suite)")
    args = parser.parse_args(argv)
    for name in args.suites or suite_names():
        try:
            info = describe(name)
        except Exception as error:
            print(f"{name}: registered, but LIBERO cannot build it ({type(error).__name__}: {error})")
            continue
        print(f"{info.name}: {len(info.tasks)} tasks, max steps {info.max_steps}")
        for task in info.tasks:
            print(f"  {task.index:3d}  {task.init_states:3d} init states  {task.language}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
