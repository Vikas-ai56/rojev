"""Subtask vocabulary of a steerable checkpoint: the `subtask` strings its low level was trained on.

Downloads each data file of the training dataset named in the checkpoint's train_config.json once (HF cache), then
reads only the `episode_index`, `task_index` and `language_persistent` columns locally. Subtask rows repeat the full
timeline on every frame of an episode, so the first frame of each episode is enough; timestamps give the order. Remote column reads were
tried and rejected: the files have ~939 row groups, so the language column is ~939 small pieces between image
columns, and a block-cached HTTP reader re-downloads the files several times. Counts are per episode, not per frame.

  uv run python extract_subtask_vocab.py --policy lerobot/pi052_robocasa
"""
import argparse
import collections
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="lerobot/pi052_robocasa")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    train = json.loads(Path(hf_hub_download(args.policy, "train_config.json")).read_text())
    dataset = train["dataset"]["repo_id"]
    tasks_df = pq.read_table(io.BytesIO(Path(hf_hub_download(dataset, "meta/tasks.parquet", repo_type="dataset"))
                                        .read_bytes())).to_pandas()
    task_of = {int(i): str(t) for t, i in zip(tasks_df.index, tasks_df["task_index"])}

    files = sorted(f for f in HfApi().list_repo_files(dataset, repo_type="dataset")
                   if f.startswith("data/") and f.endswith(".parquet"))
    episodes_with = collections.defaultdict(set)  # subtask -> episode ids
    tasks_with = collections.defaultdict(collections.Counter)  # subtask -> task string -> episodes
    seen_eps = set()
    sequences = []  # one ordered subtask timeline per episode (rows repeat on every frame; ordered by timestamp)
    for k, path in enumerate(files):
        local = hf_hub_download(dataset, path, repo_type="dataset")
        t = pq.read_table(local, columns=["episode_index", "task_index", "language_persistent"])
        last_ep = None
        for ep, ti, rows in zip(t.column("episode_index").to_pylist(), t.column("task_index").to_pylist(),
                                t.column("language_persistent").to_pylist()):
            if ep == last_ep:  # persistent rows repeat on every frame of an episode
                continue
            last_ep = ep
            seen_eps.add(ep)
            parsed = [json.loads(r) if isinstance(r, str) else r for r in rows or []]
            timeline = sorted((r for r in parsed if r.get("style") == "subtask" and r.get("content")),
                              key=lambda r: r.get("timestamp") or 0.0)
            sequences.append({"episode": int(ep), "task": task_of.get(int(ti), str(ti)),
                              "subtasks": [r["content"].strip() for r in timeline],
                              "timestamps": [r.get("timestamp") for r in timeline]})
            for r in parsed:
                if r.get("style") == "subtask" and r.get("content"):
                    s = r["content"].strip()
                    if ep not in episodes_with[s]:
                        episodes_with[s].add(ep)
                        tasks_with[s][task_of.get(int(ti), str(ti))] += 1
        print(f"[{k + 1}/{len(files)}] {path}: {len(episodes_with)} distinct subtasks so far", flush=True)

    vocab = sorted(({"subtask": s, "episodes": len(eps), "tasks": dict(tasks_with[s].most_common())}
                    for s, eps in episodes_with.items()), key=lambda v: -v["episodes"])
    out = Path(args.out or REPO / "data/vocab/pi052_robocasa_subtasks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy": args.policy, "dataset": dataset, "episodes": len(seen_eps),
                               "distinct_subtasks": len(vocab), "subtasks": vocab}, indent=1))
    seq_out = out.with_name(out.name.replace("_subtasks.json", "_sequences.jsonl"))
    seq_out.write_text("".join(json.dumps(s) + "\n" for s in sequences))
    print(f"{len(vocab)} distinct subtasks over {len(seen_eps)} episodes -> {out}")
    print(f"{len(sequences)} ordered episode timelines -> {seq_out}")


if __name__ == "__main__":
    main()
