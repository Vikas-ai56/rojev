"""MJev patch-embedding fix candidates on a T4 vs the current bf16 Conv3d (E12e).

  python patch_embed_check.py <checkpoint> <base-dir> <out.json>
"""
import time, json, torch, sys
from pathlib import Path
from speed_check import load
ckpt, base = map(Path, sys.argv[1:3])
model, _ = load(ckpt, base, torch.bfloat16, 1024)
pe = model.native.visual.patch_embed
conv = pe.proj
print(json.dumps({"conv": str(conv), "weight_dtype": str(conv.weight.dtype),
                  "cudnn": torch.backends.cudnn.is_available(), "cudnn_version": torch.backends.cudnn.version(),
                  "cudnn_benchmark": torch.backends.cudnn.benchmark}), flush=True)
x = torch.randn(2304, 1536, device="cuda:0")  # same shape MJev sends for 3 cameras
def run(fn, reps=2):
    with torch.inference_mode():
        fn(); torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps): out = fn()
        torch.cuda.synchronize()
    return out, round((time.perf_counter() - t0) * 1000 / reps, 1)
ref, t_bf16 = run(lambda: pe(x))                                  # current path (bf16 Conv3d)
conv32 = torch.nn.Conv3d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, bias=conv.bias is not None).to("cuda:0")
conv32.load_state_dict({k: v.float() for k, v in conv.state_dict().items()})
k = (-1, conv.in_channels) + tuple(conv.kernel_size)
fp32, t_fp32 = run(lambda: conv32(x.view(k)).view(-1, conv.out_channels).to(ref.dtype))
w = conv.weight.reshape(conv.out_channels, -1)                    # kernel == stride: conv = patches @ W^T + b
b = conv.bias
lin, t_lin = run(lambda: torch.nn.functional.linear(x.to(w.dtype), w, b))
lin32, t_lin32 = run(lambda: torch.nn.functional.linear(x, w.float(), None if b is None else b.float()).to(ref.dtype))
diff = lambda a: round((a.float() - ref.float()).abs().max().item(), 5)
result = {"bf16_conv3d_ms": t_bf16, "fp32_conv3d_ms": t_fp32, "bf16_linear_ms": t_lin, "fp32_linear_ms": t_lin32,
          "max_abs_diff_vs_current": {"fp32_conv3d": diff(fp32), "bf16_linear": diff(lin), "fp32_linear": diff(lin32)},
          "ref_abs_mean": round(ref.float().abs().mean().item(), 5)}
print(json.dumps(result, indent=1), flush=True)
Path(sys.argv[3]).write_text(json.dumps(result, indent=2))
