"""Compare cached PyTorch masks with the optional CUDA backend on the same GPU."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.request

import torch

ROOT = Path(__file__).resolve().parents[1]


def check_idle(args):
    if args.server_url:
        with urllib.request.urlopen(args.server_url.rstrip("/") + "/queue", timeout=10) as response:
            queue = json.load(response)
        if queue["queue_running"] or queue["queue_pending"]:
            raise RuntimeError("ComfyUI queue became busy; benchmark stopped")


def measure_cuda(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
        del result
    return {"median_ms": statistics.median(values), "samples_ms": values}


def masks(args):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_cache import load_plugin, make_attention

    class Upscale:
        @classmethod
        def execute(cls):
            return None

    plugin = load_plugin(Upscale)
    device = torch.device("cuda:0")
    assert plugin._mask_backend.available(device), "Build the native backend first"
    timings = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for size in (64, 256, 512):
            check_idle(args)
            module = make_attention(device, dtype)
            before = measure_cuda(lambda: plugin._calculate_mask_on_device(module, size, size, device, dtype), args.repeats)
            after = measure_cuda(lambda: plugin._mask_backend.make_masks(module, size, size, device, dtype), args.repeats)
            row = {"dtype": str(dtype), "size": size, "torch": before, "cuda": after,
                   "speedup": before["median_ms"] / after["median_ms"]}
            timings.append(row)
            check_idle(args)
            print(json.dumps(row), flush=True)
    peaks = {}
    module = make_attention(device)
    for name, fn in (("torch", plugin._calculate_mask_on_device), ("cuda", plugin._mask_backend.make_masks)):
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        result = fn(module, 512, 512, device, torch.float32)
        torch.cuda.synchronize()
        peaks[name] = (torch.cuda.max_memory_allocated() - start_memory) / 2**20
        del result
    return {"scope": "uncached mask generation only; CUDA events include launch gaps",
            "timings": timings, "peak_extra_mib": peaks}


def model(args):
    import numpy as np
    from PIL import Image

    if not args.comfyui_root or not args.image or not args.model:
        raise ValueError("--comfyui-root, --image and --model are required for model timing")
    core = args.comfyui_root.resolve()
    image_path = args.image.resolve()
    sys.path.insert(0, str(core))
    os.chdir(core)
    # Match the measured launcher setting without letting ComfyUI parse this script's flags.
    sys.argv = [sys.argv[0], "--fast", "fp16_accumulation"]
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader

    spec = importlib.util.spec_from_file_location("dat_cache_benchmark", ROOT / "__init__.py")
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    assert plugin.PATCH_ACTIVE
    assert plugin._mask_backend.available(torch.device("cuda:0")), "Build the native backend first"

    check_idle(args)
    image = torch.from_numpy(np.array(Image.open(image_path).convert("RGB")).astype(np.float32) / 255)[None]
    upscaler, = UpscaleModelLoader.execute(args.model)
    rows = []
    reference = None
    mask_calls = 0
    original_dispatch = plugin._mask_backend.make_masks

    def counted(*values):
        nonlocal mask_calls
        mask_calls += 1
        return original_dispatch(*values)

    plugin._mask_backend.make_masks = counted

    def run(mode, pair, warmup=False):
        nonlocal reference, mask_calls
        check_idle(args)
        plugin.clear_runtime_mask_cache()
        plugin._mask_backend.enabled = mode == "cuda"
        mask_calls = 0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_cpu, start = time.process_time(), time.perf_counter()
        result, = ImageUpscaleWithModel.execute(upscaler, image)
        torch.cuda.synchronize()
        row = {"mode": mode, "pair": pair, "warmup": warmup,
               "wall_seconds": time.perf_counter() - start,
               "cpu_seconds": time.process_time() - start_cpu,
               "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
               "misses": mask_calls, "cache_cleared": not plugin._runtime_mask_cache}
        check_idle(args)
        assert row["cache_cleared"]
        if reference is None:
            reference = result.clone()
        row["bitwise_equal"] = bool(torch.equal(reference, result))
        if not row["bitwise_equal"]:
            raise AssertionError(f"Output changed: max_abs_error={(reference - result).abs().max().item()}")
        rows.append(row)
        print(json.dumps(row), flush=True)
        return list(result.shape)

    run("torch", -1, True)
    output_shape = run("cuda", -1, True)
    orders = [("torch", "cuda") if i % 2 == 0 else ("cuda", "torch") for i in range(args.pairs)]
    random.Random(20261017).shuffle(orders)
    for pair, order in enumerate(orders):
        for mode in order:
            run(mode, pair)
    measured = [row for row in rows if not row["warmup"]]
    old = [r["wall_seconds"] for r in measured if r["mode"] == "torch"]
    new = [r["wall_seconds"] for r in measured if r["mode"] == "cuda"]
    differences = [next(r["wall_seconds"] for r in measured if r["pair"] == i and r["mode"] == "torch")
                   - next(r["wall_seconds"] for r in measured if r["pair"] == i and r["mode"] == "cuda") for i in range(args.pairs)]
    rng = np.random.default_rng(20261017)
    boot = np.median(rng.choice(differences, size=(10000, args.pairs), replace=True), axis=1)
    return {"scope": "ImageUpscaleWithModel only; model loaded before timing; two warmups excluded",
            "model": args.model, "input_shape": list(image.shape), "output_shape": output_shape,
            "torch_median_seconds": statistics.median(old), "cuda_median_seconds": statistics.median(new),
            "speedup": statistics.median(old) / statistics.median(new),
            "median_paired_saving_seconds": statistics.median(differences), "paired_savings_seconds": differences,
            "paired_median_bootstrap_95ci_seconds": np.quantile(boot, [0.025, 0.975]).tolist(),
            "all_outputs_bitwise_equal": all(row["bitwise_equal"] for row in rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("masks", "model"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--comfyui-root", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--model", help="Model filename in ComfyUI's upscale_models directory")
    parser.add_argument("--server-url", default="http://127.0.0.1:8188", help="Queue guard; pass an empty string for offline use")
    args = parser.parse_args()
    if args.repeats < 1 or args.pairs < 2:
        parser.error("--repeats must be positive and --pairs must be at least 2")
    args.output = args.output.resolve()
    check_idle(args)
    with torch.inference_mode():
        report = masks(args) if args.mode == "masks" else model(args)
    report.update({"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
                   "cuda": torch.version.cuda, "plugin_version": "1.1.0"})
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "timings")}), flush=True)


if __name__ == "__main__":
    main()
