import importlib
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import inspect
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from spandrel.architectures.DAT.__arch import DAT as dat


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "dat_layout_unit_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
layout = importlib.import_module(f"{PACKAGE}.dat_layout")
backend_module = importlib.import_module(f"{PACKAGE}.layout_backend")
CUDA_TEST = os.environ.get("DAT_CACHE_TEST_CUDA") == "1" and torch.cuda.is_available()


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.saved = []
        self.marker = getattr(dat, layout.MARKER, None)
        if self.marker is not None:
            delattr(dat, layout.MARKER)
        for class_name, method_name, _, _ in layout.METHODS:
            owner = getattr(dat, class_name)
            current = getattr(owner, method_name)
            self.saved.append((owner, method_name, current))
            setattr(owner, method_name, inspect.unwrap(current))
        layout.BACKEND.enabled = True

    def tearDown(self):
        for owner, name, original in self.saved:
            setattr(owner, name, original)
        if self.marker is not None:
            setattr(dat, layout.MARKER, self.marker)
        elif hasattr(dat, layout.MARKER):
            delattr(dat, layout.MARKER)

    def test_installation_is_local_lazy_and_idempotent(self):
        tensor_method = torch.Tensor.contiguous
        originals = [getattr(owner, name) for owner, name, _ in self.saved]
        with patch.object(layout.BACKEND, "_load", side_effect=AssertionError("must remain lazy")):
            self.assertIs(layout.install_layout_patch(), layout.BACKEND)
            first = [getattr(owner, name) for owner, name, _ in self.saved]
            self.assertIs(layout.install_layout_patch(), layout.BACKEND)
        self.assertIs(torch.Tensor.contiguous, tensor_method)
        self.assertEqual(first, [getattr(owner, name) for owner, name, _ in self.saved])
        self.assertTrue(all(a is not b for a, b in zip(originals, first)))

    def test_unknown_source_leaves_all_methods_untouched(self):
        def changed(self, x, H, W):
            return x

        dat.Adaptive_Channel_Attention.forward = changed
        before = [getattr(owner, name) for owner, name, _ in self.saved]
        with self.assertLogs(layout.LOGGER, level="WARNING"):
            self.assertIsNone(layout.install_layout_patch())
        self.assertEqual(before, [getattr(owner, name) for owner, name, _ in self.saved])
        self.assertFalse(hasattr(dat, layout.MARKER))

    def test_cpu_and_autograd_use_original_method(self):
        module = dat.SpatialGate(6).eval()
        original = dat.SpatialGate.forward
        x = torch.randn(1, 64, 12, requires_grad=True)
        expected = original(module, x, 8, 8)
        layout.install_layout_patch()
        with patch.object(layout, "_contiguous", side_effect=AssertionError("CPU fallback")):
            actual = module(x, 8, 8)
        self.assertTrue(torch.equal(expected, actual))
        actual.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_existing_forward_wrapper_is_not_overwritten(self):
        original = dat.SpatialGate.forward

        @wraps(original)
        def external_wrapper(self, x, H, W):
            return original(self, x, H, W)

        dat.SpatialGate.forward = external_wrapper
        with self.assertLogs(layout.LOGGER, level="WARNING"):
            self.assertIsNone(layout.install_layout_patch())
        self.assertIs(dat.SpatialGate.forward, external_wrapper)

    def test_static_methods_match_cpu_reference(self):
        layout.install_layout_patch()
        for batch in (1, 2):
            for height, width in ((8, 8), (9, 13)):
                x = torch.randn(batch, height * width, 48)
                modules = [
                    (dat.SpatialGate(24), layout.spatial_gate_forward),
                    (dat.Adaptive_Channel_Attention(48, num_heads=6), layout.adaptive_channel_forward),
                ]
                for block in (0, 2):
                    modules.append((dat.Adaptive_Spatial_Attention(
                        48, num_heads=6, split_size=[2, 4], shift_size=[1, 2],
                        rg_idx=0, b_idx=block,
                    ), layout.adaptive_spatial_forward))
                with torch.inference_mode():
                    for module, replacement in modules:
                        module.eval()
                        expected = module(x, height, width)
                        actual = replacement(module, x, height, width)
                        self.assertTrue(torch.equal(expected, actual), (batch, height, width, replacement.__name__))

    @unittest.skipUnless(CUDA_TEST, "CUDA test is opt-in")
    def test_cuda_dat_methods_and_training_fallback(self):
        height, width = 32, 32
        x = torch.randn(1, height * width, 180, device="cuda")
        modules = [
            dat.SpatialGate(90),
            dat.Adaptive_Channel_Attention(180, num_heads=6),
            dat.Adaptive_Spatial_Attention(180, num_heads=6, split_size=[8, 16], shift_size=[4, 8], rg_idx=0, b_idx=2),
        ]
        originals = [type(module).forward for module in modules]
        modules = [module.eval().cuda() for module in modules]
        layout.install_layout_patch()
        with torch.inference_mode():
            for module, original in zip(modules, originals):
                expected = original(module, x, height, width)
                actual = module(x, height, width)
                self.assertTrue(torch.equal(expected, actual))
        gate = modules[0].train()
        with patch.object(layout, "_contiguous", side_effect=AssertionError("training fallback")):
            with torch.no_grad():
                gate(x, height, width)
        gate.eval()
        tracked = x.clone().requires_grad_()
        with patch.object(layout, "_contiguous", side_effect=AssertionError("autograd fallback")):
            gate(tracked, height, width).sum().backward()
        self.assertIsNotNone(tracked.grad)


class BackendTests(unittest.TestCase):
    def test_cpu_fallback_does_not_load_cuda_library(self):
        backend = backend_module.LayoutBackend()
        tensor = torch.randn(1024, 180).T
        with patch.object(backend, "_load", side_effect=AssertionError("CPU must not load CUDA")):
            actual = backend.contiguous(tensor)
        self.assertTrue(torch.equal(actual, tensor.contiguous()))

    def test_unsupported_platform_remains_optional(self):
        backend = backend_module.LayoutBackend()
        with patch.object(backend_module.sys, "platform", "linux"):
            self.assertIsNone(backend._load())
        self.assertTrue(backend._attempted)

    def test_missing_library_warns_once_and_remains_optional(self):
        backend = backend_module.LayoutBackend()
        with patch.object(backend_module.sys, "platform", "win32"), patch.object(torch.version, "cuda", "13.0"):
            with patch.object(backend_module, "CublasTranspose", side_effect=OSError("test library missing")) as constructor:
                with self.assertLogs(backend_module.LOGGER, level="WARNING"):
                    self.assertIsNone(backend._load())
                self.assertIsNone(backend._load())
        constructor.assert_called_once()

    @unittest.skipUnless(CUDA_TEST, "CUDA test is opt-in")
    def test_cuda_strides_offsets_streams_and_fp32_values(self):
        backend = backend_module.LayoutBackend()
        self.assertIsNotNone(backend._load(), "Windows CUDA test requires bundled cuBLAS")
        for stream in (torch.cuda.default_stream(), torch.cuda.Stream()):
            with torch.cuda.stream(stream), torch.inference_mode():
                for channels, multiplier, reverse in ((180, 3, False), (180, 2, False), (180, 1, True), (90, 1, True)):
                    base = torch.randn((1, 4096, channels * multiplier), device="cuda")
                    tensor = base[:, :, -channels:].transpose(1, 2)
                    if reverse:
                        tensor = tensor.contiguous().transpose(1, 2)
                    self.assertTrue(backend_module.eligible(tensor))
                    self.assertIsNotNone(backend._backend.copy(tensor), "cuBLAS must execute rather than fall back")
                    result = backend.contiguous(tensor)
                    self.assertTrue(result.is_contiguous())
                    self.assertTrue(torch.equal(result.view(torch.int32), tensor.contiguous().view(torch.int32)))
                bits = torch.tensor([0, -2147483648, 1, -2147483647, 8388607, 8388608, 2139095040, -8388608, 2143289345], dtype=torch.int32, device="cuda")
                tensor = bits.repeat(8192).reshape(256, 288).view(torch.float32).T
                result = backend.contiguous(tensor)
                reference = tensor.contiguous()
                torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)
                finite = torch.isfinite(reference)
                self.assertTrue(torch.equal(result[finite].view(torch.int32), reference[finite].view(torch.int32)))
            stream.synchronize()

    @unittest.skipUnless(CUDA_TEST, "CUDA test is opt-in")
    def test_cuda_fallbacks_and_error_propagation(self):
        backend = backend_module.LayoutBackend()
        tensor = torch.randn(1024, 180, device="cuda").T
        with patch.object(backend, "_load", side_effect=AssertionError("fallback")):
            backend.contiguous(tensor)  # grad-enabled path
            with torch.inference_mode():
                backend.enabled = False
                backend.contiguous(tensor)
                backend.enabled = True
                with patch.dict(os.environ, {"DAT_CACHE_LAYOUT": "0"}):
                    backend.contiguous(tensor)
                for other in (tensor.half(), tensor.bfloat16(), tensor.double(), tensor[:8, :8], tensor.unsqueeze(0).expand(2, -1, -1)):
                    self.assertFalse(backend_module.eligible(other))
                    self.assertTrue(torch.equal(backend.contiguous(other), other.contiguous()))
        with torch.inference_mode(), patch.object(backend, "_load", return_value=SimpleNamespace(copy=lambda x: None)):
            self.assertTrue(torch.equal(backend.contiguous(tensor), tensor.contiguous()))

        def fail(x):
            raise RuntimeError("expected CUDA error")

        with torch.inference_mode(), patch.object(backend, "_load", return_value=SimpleNamespace(copy=fail)):
            with self.assertRaisesRegex(RuntimeError, "expected CUDA error"):
                backend.contiguous(tensor)

    @unittest.skipUnless(CUDA_TEST, "CUDA test is opt-in")
    def test_cuda_graph_capture_falls_back_to_torch(self):
        backend = backend_module.LayoutBackend()
        tensor = torch.randn(1024, 180, device="cuda").T
        with torch.inference_mode():
            backend.contiguous(tensor)
            graph = torch.cuda.CUDAGraph()
            with patch.object(backend._backend, "geam", side_effect=AssertionError("capture must use PyTorch")):
                with torch.cuda.graph(graph):
                    result = backend.contiguous(tensor)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(result, tensor.contiguous()))

    @unittest.skipUnless(CUDA_TEST, "CUDA test is opt-in")
    def test_concurrent_threads_use_their_own_streams(self):
        backend = backend_module.LayoutBackend()

        def worker(seed):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream), torch.inference_mode():
                tensor = (torch.arange(180 * 1024, device="cuda", dtype=torch.float32) + seed).reshape(1024, 180).T
                actual = backend.contiguous(tensor)
                equal = bool(torch.equal(actual, tensor.contiguous()))
            stream.synchronize()
            return equal

        with ThreadPoolExecutor(max_workers=2) as executor:
            self.assertTrue(all(executor.map(worker, (1, 17))))


if __name__ == "__main__":
    unittest.main()
