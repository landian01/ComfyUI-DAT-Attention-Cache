"""Full DAT node A/B with identical FP32 math and mask caching in both modes."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import urllib.request


GPU_QUERY = "temperature.gpu,clocks.current.sm,utilization.gpu,clocks_event_reasons.sw_thermal_slowdown"


def read_gpu_status(uuid):
    result = subprocess.run(
        ["nvidia-smi", "--id", uuid, "--query-gpu=" + GPU_QUERY, "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return parse_gpu_status(result.stdout)


def parse_gpu_status(line):
    temperature, clock, utilization, thermal = (part.strip() for part in line.strip().split(","))
    return {"temperature_c": int(temperature), "sm_clock_mhz": int(clock),
            "utilization_percent": int(utilization), "thermal_slowdown": thermal == "Active"}


class GpuTelemetry:
    def __init__(self, uuid):
        self.uuid = uuid
        self.samples = []
        self.ready = threading.Event()

    def start(self):
        self.process = subprocess.Popen(
            ["nvidia-smi", "--id", self.uuid, "--query-gpu=" + GPU_QUERY,
             "--format=csv,noheader,nounits", "--loop-ms=1000"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        def collect():
            for line in self.process.stdout:
                self.samples.append(parse_gpu_status(line))
                self.ready.set()
        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()
        if not self.ready.wait(10):
            self.stop()
            raise RuntimeError("Could not start selected-GPU telemetry")

    def stop(self):
        self.process.terminate()
        self.process.wait(timeout=10)
        self.thread.join(timeout=10)
        self.process.stdout.close()

    def summary(self):
        active = [sample for sample in self.samples if sample["utilization_percent"] >= 80]
        return {
            "samples": len(self.samples),
            "max_temperature_c": max(sample["temperature_c"] for sample in self.samples),
            "active_sm_clock_range_mhz": [min(sample["sm_clock_mhz"] for sample in active), max(sample["sm_clock_mhz"] for sample in active)] if active else None,
            "thermal_slowdown_seen": any(sample["thermal_slowdown"] for sample in self.samples),
            "thermal_slowdown_samples": sum(sample["thermal_slowdown"] for sample in self.samples),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Filename in ComfyUI's upscale_models directory")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-name", required=True, help="Required substring in the selected GPU name")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--cooldown-temperature", type=int, default=60)
    parser.add_argument("--cooldown-min-seconds", type=int, default=30)
    parser.add_argument("--queue-url", help="Abort if this ComfyUI queue is busy or cannot be checked")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly --gpu-uuid before starting Python")
    if args.pairs < 2:
        raise ValueError("Use at least two counterbalanced A/B pairs")

    def check_queue_idle():
        if args.queue_url:
            with urllib.request.urlopen(args.queue_url, timeout=5) as response:
                queue = json.load(response)
            if queue["queue_running"] or queue["queue_pending"]:
                raise RuntimeError("ComfyUI queue became busy; discard this benchmark attempt")

    check_queue_idle()
    os.environ["DAT_CACHE_NATIVE"] = "0"
    os.environ.pop("DAT_CACHE_LAYOUT", None)

    import numpy as np
    from PIL import Image
    import torch

    if torch.cuda.device_count() != 1 or args.gpu_name not in torch.cuda.get_device_name(0):
        raise RuntimeError("Unexpected visible GPU; no benchmark will be run")
    sys.path.insert(0, str(args.comfyui.resolve()))
    os.chdir(args.comfyui)
    sys.argv = [sys.argv[0], "--fast", "fp16_accumulation"]
    from comfy import model_management
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("dat_layout_benchmark", root / "__init__.py")
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    if not plugin.PATCH_ACTIVE or not plugin.LAYOUT_ACTIVE:
        raise RuntimeError("Both DAT cache and layout patches must be installed")
    backend = plugin.LAYOUT_BACKEND
    model, = UpscaleModelLoader.execute(args.model)
    parameter_dtypes = {str(parameter.dtype) for parameter in model.model.parameters()}
    if parameter_dtypes != {"torch.float32"}:
        raise RuntimeError(f"Expected FP32 model, got {parameter_dtypes}")
    model_management.load_models_gpu([model.patcher], memory_required=5 * 2**30, force_full_load=True)
    pixels = np.array(Image.open(args.image).convert("RGB")).astype(np.float32) / 255
    image = torch.from_numpy(pixels)[None]
    original_contiguous = torch.Tensor.contiguous
    rows = []
    reference = None
    tile_shapes = []

    class GuardedModel:
        def __getattr__(self, name):
            return getattr(model, name)

        def __call__(self, tile):
            check_queue_idle()
            tile_shapes.append(list(tile.shape))
            result = model(tile)
            check_queue_idle()
            return result

    guarded_model = GuardedModel()

    def run(mode, pair, warmup=False):
        nonlocal reference
        backend.enabled = mode == "layout"
        plugin.clear_runtime_mask_cache()
        torch.cuda.synchronize()
        cooldown_start = time.perf_counter()
        while True:
            check_queue_idle()
            status = read_gpu_status(args.gpu_uuid)
            cooled_for = time.perf_counter() - cooldown_start
            if status["temperature_c"] <= args.cooldown_temperature and cooled_for >= args.cooldown_min_seconds:
                break
            if cooled_for > 180:
                raise RuntimeError(f"GPU did not cool to target temperature: {status}")
            time.sleep(2)
        print(json.dumps({"starting_mode": mode, "pair": pair, "warmup": warmup, "start_gpu_status": status}), flush=True)
        telemetry = GpuTelemetry(args.gpu_uuid)
        telemetry.start()
        tile_shapes.clear()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            check_queue_idle()
            result, = ImageUpscaleWithModel.execute(guarded_model, image)
            torch.cuda.synchronize()
            check_queue_idle()
            wall_seconds = time.perf_counter() - started
        finally:
            telemetry.stop()
        row = {
            "mode": mode, "pair": pair, "warmup": warmup,
            "wall_seconds": wall_seconds,
            "start_gpu_status": status, "gpu_telemetry": telemetry.summary(),
            "tile_shapes": list(tile_shapes),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "cache_cleared": not plugin._runtime_mask_cache and not plugin._runtime_shape,
            "output_device": str(result.device),
            "output_shape": list(result.shape),
            "output_dtype": str(result.dtype),
        }
        if reference is None:
            reference = result.clone()
        row["bitwise_equal"] = bool(torch.equal(reference.view(torch.int32), result.view(torch.int32)))
        row["finite"] = bool(torch.isfinite(result).all())
        row["max_abs_error"] = 0.0 if row["bitwise_equal"] else float((reference - result).abs().max())
        if not row["bitwise_equal"] or not row["finite"] or not row["cache_cleared"]:
            raise RuntimeError(f"DAT parity/cache gate failed: {row}")
        if torch.Tensor.contiguous is not original_contiguous:
            raise RuntimeError("Global Tensor.contiguous was modified")
        print(json.dumps(row), flush=True)
        return row

    try:
        with torch.inference_mode():
            warmups = [run(mode, -1, warmup=True) for mode in ("torch", "layout")]
            if backend._backend is None:
                raise RuntimeError("cuBLAS layout backend did not activate")
            for pair in range(args.pairs):
                order = ("torch", "layout") if pair % 2 == 0 else ("layout", "torch")
                for mode in order:
                    rows.append(run(mode, pair))
                    args.output.with_suffix(".progress.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        before = statistics.median(row["wall_seconds"] for row in rows if row["mode"] == "torch")
        after = statistics.median(row["wall_seconds"] for row in rows if row["mode"] == "layout")
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "plugin_version": plugin.PATCH_VERSION,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "visible_gpu_count": torch.cuda.device_count(),
            "scope": "complete ImageUpscaleWithModel call, including all tiles, transfers and stitching",
            "model": args.model, "input_shape": list(image.shape),
            "precision": "FP32", "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": sys.version.split()[0], "spandrel": importlib.metadata.version("spandrel"),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "mask_cache_both_modes": True, "native_mask_both_modes": False,
            "global_tensor_contiguous_unchanged": torch.Tensor.contiguous is original_contiguous,
            "cooldown_temperature_c": args.cooldown_temperature,
            "cooldown_min_seconds": args.cooldown_min_seconds,
            "cooldown_excluded_from_node_timing": True,
            "comfyui_queue_idle_guard": bool(args.queue_url),
            "torch_median_seconds": before, "layout_median_seconds": after,
            "speedup": before / after, "time_reduction_percent": (1 - after / before) * 100,
            "warmups": warmups, "rows": rows,
        }
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key not in ("rows", "warmups")}), flush=True)
    finally:
        backend.enabled = True
        plugin.clear_runtime_mask_cache()


if __name__ == "__main__":
    main()
