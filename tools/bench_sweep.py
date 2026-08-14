"""Find the (config, batch) combos that actually fit in VRAM, and how fast.

Uses random tensors -- the real pipeline is GPU-resident and adds ~0 per-step
cost, so model fwd/bwd is what we need to measure. Anything that spills into
WDDM shared memory shows up as a throughput cliff, not an OOM, so watch the
it/s column as well as the GiB column.

Run:  .venv/Scripts/python -u tools/bench_sweep.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nafnet_sr import build_model, count_params  # noqa: E402
from src.losses import RestorationLoss  # noqa: E402

dev = 'cuda'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

TOTAL = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
print(f'{torch.cuda.get_device_name(0)}  {TOTAL:.1f} GiB\n')
print('%-6s %6s %10s %9s %9s %11s' % ('config', 'batch', 'params(M)', 'it/s', 'peak GiB', '100k iters'))

for cfg in ('s', 'm'):
    for bs in (8, 16, 24, 32):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            model = build_model(cfg).to(dev).to(memory_format=torch.channels_last)
            crit = RestorationLoss().to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.9), weight_decay=0.0)
            x = torch.rand(bs, 1, 128, 128, device=dev).to(memory_format=torch.channels_last)
            y = torch.rand(bs, 1, 256, 256, device=dev)

            for _ in range(6):                     # warmup + cudnn autotune
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, _ = crit(model(x), y)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            n = 30
            t0 = time.perf_counter()
            for _ in range(n):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, _ = crit(model(x), y)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / n

            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            print('%-6s %6d %10.2f %9.2f %9.2f %10.1fh'
                  % (cfg, bs, count_params(model) / 1e6, 1 / dt, peak, 100000 * dt / 3600))
        except torch.cuda.OutOfMemoryError:
            print('%-6s %6d %10s %9s %9s %11s' % (cfg, bs, '-', 'OOM', '-', '-'))
        finally:
            del model, opt, crit, x, y
            torch.cuda.empty_cache()

print('\nInference-only (what the throughput score measures):')
print('%-6s %6s %9s %14s' % ('config', 'batch', 'peak GiB', 'ms/image'))
for cfg in ('s', 'm'):
    for bs in (32, 64):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(cfg).to(dev).eval().to(memory_format=torch.channels_last)
        x = torch.rand(bs, 1, 128, 128, device=dev).to(memory_format=torch.channels_last)
        with torch.inference_mode():
            for _ in range(6):
                with torch.autocast('cuda', dtype=torch.float16):
                    model(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                with torch.autocast('cuda', dtype=torch.float16):
                    model(x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 30
        print('%-6s %6d %9.2f %14.3f'
              % (cfg, bs, torch.cuda.max_memory_allocated() / 2 ** 30, 1000 * dt / bs))
        del model, x
        torch.cuda.empty_cache()
