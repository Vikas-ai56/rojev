"""Rollout sampling and the whole-attempt case plan on synthetic ArmnetBench episode rows (no network): only pi0.5
successes and failures enter the pool, the sample is label-balanced and balanced within each task where the task has
both labels, a seed reproduces it, and every case shows the attempt's last frame per view (the newest frame)."""
from collections import Counter
from dataclasses import asdict

from rollouts import TRUTH, VIEWS, Attempt, attempts, choose, frame_path, plan

POOLS = {"stack the blocks": (20, 20), "clip the cable": (3, 30), "insert the ring": (25, 5)}


def synthetic_rows() -> list[dict]:
    rows, episode = [], 0
    for task, (successes, failures) in POOLS.items():
        for label, n in (("successful", successes), ("failure", failures)):
            for _ in range(n):
                rows.append({"episode_index": episode, "tasks": [task], "success_class": label,
                             "policy_type": "pi0.5"})
                episode += 1
    rows += [{"episode_index": episode, "tasks": ["stack the blocks"], "success_class": "suboptimal",
              "policy_type": "pi0.5"},
             {"episode_index": episode + 1, "tasks": ["stack the blocks"], "success_class": "successful",
              "policy_type": "teleoperated"},
             {"episode_index": episode + 2, "tasks": ["stack the blocks"], "success_class": "failure",
              "policy_type": "act"}]
    return rows


def per_task(chosen: list[Attempt]) -> Counter:
    return Counter((a.instruction, a.label) for a in chosen)


def test_pool_keeps_only_pi05_successes_and_failures():
    pool = attempts(synthetic_rows())
    assert len(pool) == sum(s + f for s, f in POOLS.values())
    assert {a.label for a in pool} == set(TRUTH)


def test_sample_balances_labels_and_each_task_when_the_pool_is_deep():
    chosen = choose(attempts(synthetic_rows()), target=40, seed=0)
    counts = per_task(chosen)
    assert Counter(a.label for a in chosen) == {"successful": 20, "failure": 20}
    assert all(counts[task, "successful"] == counts[task, "failure"] for task in POOLS)
    assert counts["clip the cable", "successful"] == 3 and counts["insert the ring", "successful"] == 5
    assert len({a.episode for a in chosen}) == 40


def test_a_short_pool_gives_every_attempt_of_the_scarcer_label_and_spreads_the_other():
    chosen = choose(attempts(synthetic_rows()), target=300, seed=0)
    counts = per_task(chosen)
    assert Counter(a.label for a in chosen) == {"successful": 48, "failure": 48}
    assert [counts[task, "successful"] for task in POOLS] == [20, 3, 25]
    assert [counts[task, "failure"] for task in POOLS] == [20, 23, 5]


def test_a_seed_reproduces_the_sample_and_another_seed_changes_it():
    pool = attempts(synthetic_rows())
    assert choose(pool, 40, seed=7) == choose(pool, 40, seed=7)
    assert choose(pool, 40, seed=7) != choose(pool, 40, seed=8)


def test_cases_show_the_last_frame_of_every_view():
    pool = attempts(synthetic_rows())
    chosen = [asdict(a) for a in choose(pool, 40, seed=0)]
    lengths = {a["episode"]: 100 + 7 * a["episode"] for a in chosen}
    cases = plan(chosen, lengths)
    assert len(cases) == 40 and len({c["id"] for c in cases}) == 40
    for case, attempt in zip(cases, chosen):
        last = lengths[case["episode"]] - 1
        assert case["kind"] == "status" and case["source"] == "armnetbench" and case["frame"] == last
        assert case["goal"] == case["step"] == attempt["instruction"]
        assert case["truth"] == {"successful": "done", "failure": "failed"}[attempt["label"]]
        assert set(case["images"]) == set(VIEWS) == {"top", "wrist", "overhead"}
        for view, paths in case["images"].items():
            assert paths == [frame_path(case["episode"], view, last)]
    assert {c["truth"] for c in cases} == {"done", "failed"}
