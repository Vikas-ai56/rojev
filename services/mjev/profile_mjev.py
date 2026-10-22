"""Time one MJev request by stage and with torch.profiler (E12c).

  python profile_mjev.py <checkpoint> <base-dir> <testdata>
"""
import time, json, functools, torch, sys
from pathlib import Path
from torch.profiler import profile, ProfilerActivity
from speed_check import load
from check_precision import questions, scenes
from serve_mjev import ask
ckpt, base, testdata = map(Path, sys.argv[1:4])
model, builder = load(ckpt, base, torch.bfloat16, 1024)
data = scenes(testdata); images, subtasks = next(iter(data.values()))
ask(model, builder, "cuda:0", images, questions(subtasks))  # warm-up
stages = {}
def timed(obj, attr, label):
    fn = getattr(obj, attr)
    @functools.wraps(fn)
    def w(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize(); stages[label] = stages.get(label, 0) + (time.perf_counter() - t0) * 1000
        return out
    setattr(obj, attr, w)
timed(builder, "encode_specs", "cpu_preprocess(encode_specs)")
timed(model, "embeddings", "vision+embeddings")
timed(model.native.language_model, "forward", "language_trunk")
timed(model.native.language_model.native_qk, "forward", "qk_readout")
timed(model, "forward", "model_forward_total")
t0 = time.perf_counter(); ask(model, builder, "cuda:0", images, questions(subtasks)); total = (time.perf_counter() - t0) * 1000
print(json.dumps({"request_total_ms": round(total, 1), **{k: round(v, 1) for k, v in stages.items()}}, indent=1), flush=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    ask(model, builder, "cuda:0", images, questions(subtasks))
print(p.key_averages().table(sort_by="cpu_time_total", row_limit=18), flush=True)
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=12), flush=True)
