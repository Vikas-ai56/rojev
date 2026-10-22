"""Export one runtime run to Phoenix as a span tree, after the run (nothing traced on the control path).

Input is plain records (dicts), so any executor can export: owner events (`OwnerRecord.events`), S1 requests,
commitments and S2 decisions (per-call logs), deviations, and the run's config and summary. Times are the robot clock
in seconds from the run start; `t0_ns` maps them to wall time.

  run                          config + headline metrics
  ├── command <id> <op> …      from issue to the next command (or run end); events: ACKs, review, keyframe, deviation
  ├── s1.request               capture → submit; subtask, samples, commitment (floor, steps, reason, votes)
  ├── s2.<lane>.decide         request → proposal; trigger, operation, reason, outcome (applied / refused + why)
  └── user                     each instruction the user gave (zero-length)
"""
import json

from opentelemetry import trace


def _ns(t0_ns: int, t: float) -> int:
    return t0_ns + int(t * 1e9)


def _attrs(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        key = f"{prefix}{k}"
        out[key] = v if isinstance(v, (str, bool, int, float)) else json.dumps(v, default=str)
    return out


def export_run(tracer: trace.Tracer, name: str, t0_ns: int, duration_s: float, config: dict, summary: dict,
               owner_events: list[dict], s1_requests: list[dict], commits: list[dict],
               decisions: list[dict]) -> None:
    root = tracer.start_span(name, start_time=t0_ns, attributes=_attrs(config, "config.") | _attrs(summary, "summary."))
    ctx = trace.set_span_in_context(root)
    _commands(tracer, ctx, t0_ns, duration_s, owner_events)
    _s1(tracer, ctx, t0_ns, s1_requests, commits)
    _s2(tracer, ctx, t0_ns, decisions, owner_events)
    for e in owner_events:
        if e["kind"] == "instruction":
            span = tracer.start_span("user", context=ctx, start_time=_ns(t0_ns, e["t"]),
                                     attributes={"text": e["text"], "instruction_rev": e["rev"]})
            span.end(end_time=_ns(t0_ns, e["t"]))
    root.end(end_time=_ns(t0_ns, duration_s))


def _commands(tracer, ctx, t0_ns: int, duration_s: float, events: list[dict]) -> None:
    issued = [e for e in events if e["kind"] in ("command", "reaffirm")]
    for i, e in enumerate(issued):
        end = issued[i + 1]["t"] if i + 1 < len(issued) else duration_s
        span = tracer.start_span(f"command {e['cmd']} {e['op']} {e.get('subtask') or ''}".strip(), context=ctx,
                                 start_time=_ns(t0_ns, e["t"]), attributes=_attrs(e))
        for x in events:
            if x.get("cmd") == e["cmd"] and x["kind"] in ("ack", "review", "keyframe", "deviation", "stale deviation"):
                span.add_event(x["kind"] if x["kind"] != "ack" else f"ack {x['status']}", attributes=_attrs(x),
                               timestamp=_ns(t0_ns, x["t"]))
        span.end(end_time=_ns(t0_ns, max(end, e["t"])))


def _s1(tracer, ctx, t0_ns: int, requests: list[dict], commits: list[dict]) -> None:
    served = [r for r in requests if "error" not in r]
    for r in requests:
        attrs = _attrs({k: v for k, v in r.items() if k not in ("t_start", "t_end")})
        if "error" not in r:
            index = served.index(r)
            if index < len(commits):
                attrs |= _attrs(commits[index], "commit.")
        span = tracer.start_span("s1.request", context=ctx, start_time=_ns(t0_ns, r["t_start"]), attributes=attrs)
        if "error" in r:
            span.set_status(trace.Status(trace.StatusCode.ERROR, r["error"]))
        span.end(end_time=_ns(t0_ns, r["t_end"]))


def _s2(tracer, ctx, t0_ns: int, decisions: list[dict], events: list[dict]) -> None:
    outcome = {}
    for e in events:
        if "request" in e and e["kind"] in ("command", "reaffirm", "continue", "refused"):
            outcome[e["request"]] = {"outcome": "refused" if e["kind"] == "refused" else f"applied: {e['kind']}",
                                     "why": e.get("why")}
    for d in decisions:
        attrs = _attrs({k: v for k, v in d.items() if k not in ("t_start", "t_end")}) | _attrs(
            outcome.get(d["request"], {"outcome": "not delivered"}))
        span = tracer.start_span(f"s2.{d['lane']}.decide", context=ctx, start_time=_ns(t0_ns, d["t_start"]),
                                 attributes=attrs)
        if "error" in d:
            span.set_status(trace.Status(trace.StatusCode.ERROR, d["error"]))
        span.end(end_time=_ns(t0_ns, d["t_end"]))
