"""Fit the execution monitor on nominal runs recorded with `realtime.py --monitor-record` (no faults injected).

  uv run python calibrate_monitor.py runs/*.npz --window-s 0.5 --segment-s 5 --alpha 0.1 --out monitor.json
Prints the fitted gain, the threshold, the number of calibration segments and the false-alarm bound per minute.
Runs must be like the runs the monitor will watch: a run whose loop fell behind real time (`--min-realtime-factor`) or
whose recorded policy differs from the others is refused, never silently mixed in.
"""
import argparse
from pathlib import Path

from embodiment import PANDA_OMRON, PandaOmronEffect
from rojev.system1 import Recording, fit_monitor


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", nargs="+", type=Path)
    p.add_argument("--window-s", type=float, default=0.5)
    p.add_argument("--segment-s", type=float, default=5.0)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--min-realtime-factor", type=float, default=0.98)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    hz = PANDA_OMRON.control_hz
    runs = [Recording.load(r) for r in a.recordings]
    for path, run in zip(a.recordings, runs):
        rtf = run.conditions.get("realtime_factor")
        if rtf is None or rtf < a.min_realtime_factor:
            p.error(f"{path}: real-time factor {rtf} below {a.min_realtime_factor} (or not recorded); re-record it")
    policies = {run.conditions.get("policy") for run in runs}
    if len(policies) != 1:
        p.error(f"recordings mix policies {sorted(map(str, policies))}")
    model = fit_monitor(runs, round(a.window_s * hz), round(a.segment_s * hz), a.alpha, PandaOmronEffect().resolution)
    a.out.write_text(model.to_json())
    print(f"gain (command → tracked):\n{model.gain.round(4)}\nscale {model.scale.round(5)}")
    print(f"exercised command regressors (window, previous window): {model.exercised.astype(int)}")
    print(f"threshold {model.threshold:.3f} from {model.calibration_segments} segments; "
          f"≤ {model.alarms_per_minute(hz):.2f} nominal false alarms / min")
    print(f"[calibrate] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
