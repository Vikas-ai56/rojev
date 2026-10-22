"""Build the precision-check inputs: random RoboCasa pretrain scenes + the subtasks arm D would ask MJev about.

N tasks sampled with a fixed seed (no hand-picking); for each: the 3 camera frames at reset and the top-M shortlist
from the candidate filter for its instruction (what `rojev/supervisor/onejev_arm.py` judges at the first decision).
Runs in the executor env (needs RoboCasa + rojev):

  cd executors/robocasa-pi052 && uv run python ../../services/mjev/make_testdata.py --n 4 --seed 0
Writes services/mjev/testdata/<task>/{<camera>.png, meta.json}; replaces any previous testdata.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from lerobot.envs.robocasa import RoboCasaEnv
from PIL import Image
from robocasa.utils.dataset_registry import PRETRAINING_TASKS

from rojev.legacy_v2 import CandidateBuilder

HERE = Path(__file__).resolve().parent
SEQUENCES = HERE.parents[1] / "data/vocab/pi052_robocasa_sequences.jsonl"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--m", type=int, default=3, help="shortlist subtasks judged per scene")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    timelines = [json.loads(line) for line in SEQUENCES.read_text().splitlines()]
    builder = CandidateBuilder.fit([(e["task"], e["subtasks"]) for e in timelines if e["subtasks"]])
    out = HERE / "testdata"
    shutil.rmtree(out, ignore_errors=True)
    for task in random.Random(a.seed).sample(sorted(PRETRAINING_TASKS["pretrain300"]), a.n):
        env = RoboCasaEnv(task=task, split="pretrain", obj_registries=("objaverse", "lightwheel"),
                          terminate_on_success=False)
        obs, _ = env.reset(seed=1000)
        (out / task).mkdir(parents=True)
        for cam, img in obs["pixels"].items():
            Image.fromarray(img).save(out / task / f"{cam}.png")
        meta = {"task": task, "instruction": env.task_description,
                "subtasks": builder.shortlist(env.task_description, [], a.m), "seed": a.seed}
        (out / task / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta), flush=True)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
