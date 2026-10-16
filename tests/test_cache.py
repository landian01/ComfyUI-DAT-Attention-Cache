from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from spandrel.architectures.DAT.__arch.DAT import Adaptive_Spatial_Attention as Attention

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_MASK = Attention.calculate_mask
MISSING = object()
ATTRIBUTES = ("calculate_mask", "get_runtime_mask", "clear_runtime_mask_cache", "_comfy_dat_mask_cache_patch")


def make_attention(device="cpu", dtype=torch.float32, split=(8, 16), shift=(4, 8)):
    module = torch.nn.Module()
    module.register_parameter("weight", torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype)))
    module.split_size = split
    module.shift_size = shift
    return module


@contextmanager
def comfyui_stub(upscale_class):
    parent = ModuleType("comfy_extras")
    parent.__path__ = []
    core = ModuleType("comfy_extras.nodes_upscale_model")
    core.ImageUpscaleWithModel = upscale_class
    with patch.dict(sys.modules, {"comfy_extras": parent, "comfy_extras.nodes_upscale_model": core}):
        yield


def load_plugin(upscale_class):
    spec = importlib.util.spec_from_file_location("dat_cache_plugin_test", ROOT / "__init__.py")
    plugin = importlib.util.module_from_spec(spec)
    with comfyui_stub(upscale_class):
        spec.loader.exec_module(plugin)
    return plugin


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.originals = {name: Attention.__dict__.get(name, MISSING) for name in ATTRIBUTES}

        class Upscale:
            @classmethod
            def execute(cls, *args, **kwargs):
                return args, kwargs

        self.upscale = Upscale
        self.plugin = None

    def tearDown(self):
        if self.plugin is not None:
            self.plugin.clear_runtime_mask_cache()
        for name, value in self.originals.items():
            if value is MISSING:
                if name in Attention.__dict__:
                    delattr(Attention, name)
            else:
                setattr(Attention, name, value)

    def install(self):
        self.plugin = load_plugin(self.upscale)
        self.assertTrue(self.plugin.PATCH_ACTIVE)
        return self.plugin

    def mask(self, module=None, height=32, width=48, dtype=torch.float32):
        return Attention.calculate_mask(module or make_attention(), height, width, dtype=dtype)

    def test_cpu_masks_match_spandrel_exactly(self):
        plugin = self.install()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for split, shift in (((8, 16), (4, 8)), ((16, 8), (8, 4))):
                with self.subTest(dtype=dtype, split=split):
                    module = make_attention(dtype=dtype, split=split, shift=shift)
                    expected = ORIGINAL_MASK(module, 32, 48, dtype=dtype)
                    actual = self.mask(module, dtype=dtype)
                    for result, reference in zip(actual, expected):
                        self.assertTrue(torch.equal(result, reference))
                    plugin.clear_runtime_mask_cache()

    def test_dtype_defaults_to_module_dtype(self):
        self.install()
        result = Attention.calculate_mask(make_attention(dtype=torch.float16), 32, 48)
        self.assertEqual(result[0].dtype, torch.float16)

    def test_same_geometry_is_shared_across_modules(self):
        self.install()
        first = self.mask(make_attention())
        second = self.mask(make_attention())
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])

    def test_dtype_is_part_of_cache_key(self):
        self.install()
        first = self.mask(dtype=torch.float32)
        second = self.mask(dtype=torch.float16)
        self.assertIsNot(first[0], second[0])
        self.assertEqual(second[0].dtype, torch.float16)

    def test_shape_change_discards_previous_shape(self):
        plugin = self.install()
        first = self.mask()
        self.mask(height=48, width=64)
        second = self.mask()
        self.assertIsNot(first[0], second[0])
        self.assertEqual(len(plugin._runtime_mask_cache[("cpu", None)]), 1)

    def test_window_configuration_is_part_of_cache_key(self):
        self.install()
        first = self.mask(make_attention(shift=(4, 8)))
        second = self.mask(make_attention(shift=(2, 4)))
        self.assertIsNot(first[0], second[0])
        self.assertFalse(torch.equal(first[0], second[0]))

    def test_lru_cache_is_bounded_and_evicts_oldest(self):
        plugin = self.install()
        modules = [make_attention(shift=(1 + i % 7, 1 + i // 7)) for i in range(9)]
        first = self.mask(modules[0])
        for module in modules[1:]:
            self.mask(module)
        self.assertEqual(len(plugin._runtime_mask_cache[("cpu", None)]), plugin.MAX_VARIANTS_PER_DEVICE)
        self.assertIsNot(first[0], self.mask(modules[0])[0])

    def test_execute_and_alias_release_cache_and_preserve_return(self):
        plugin = self.install()
        for execute in (self.upscale.execute, self.upscale.upscale):
            self.mask()
            self.assertEqual(execute("image", option=3), (("image",), {"option": 3}))
            self.assertFalse(plugin._runtime_mask_cache)
            self.assertFalse(plugin._runtime_shape)

    def test_exception_releases_cache_and_propagates(self):
        def fail(cls):
            raise RuntimeError("expected test failure")

        self.upscale.execute = classmethod(fail)
        plugin = self.install()
        self.mask()
        with self.assertRaisesRegex(RuntimeError, "expected test failure"):
            self.upscale.execute()
        self.assertFalse(plugin._runtime_mask_cache)
        self.assertFalse(plugin._runtime_shape)

    def test_native_runtime_cache_is_used_without_replacing_mask(self):
        cleared = []
        Attention.get_runtime_mask = lambda self: None
        Attention.clear_runtime_mask_cache = classmethod(lambda cls: cleared.append(True))
        self.install()
        self.assertIs(Attention.calculate_mask, ORIGINAL_MASK)
        self.upscale.execute()
        self.assertEqual(cleared, [True])

    def test_installation_is_idempotent(self):
        plugin = self.install()
        mask = Attention.calculate_mask
        execute = self.upscale.execute.__func__
        with comfyui_stub(self.upscale):
            self.assertTrue(plugin._install_patch())
        self.assertIs(Attention.calculate_mask, mask)
        self.assertIs(self.upscale.execute.__func__, execute)

    def test_unsupported_cleanup_does_not_install_mask_cache(self):
        self.upscale.execute = lambda self: None
        with self.assertLogs("dat_cache_plugin_test", level="ERROR"):
            self.plugin = load_plugin(self.upscale)
        self.assertFalse(self.plugin.PATCH_ACTIVE)
        self.assertIs(Attention.calculate_mask, ORIGINAL_MASK)

    def test_unsupported_mask_signature_is_left_unchanged(self):
        unsupported = lambda self, height, width: None
        Attention.calculate_mask = unsupported
        with self.assertLogs("dat_cache_plugin_test", level="ERROR"):
            self.plugin = load_plugin(self.upscale)
        self.assertFalse(self.plugin.PATCH_ACTIVE)
        self.assertIs(Attention.calculate_mask, unsupported)

    @unittest.skipUnless(os.environ.get("DAT_CACHE_TEST_CUDA") == "1" and torch.cuda.is_available(), "CUDA test is opt-in")
    def test_cuda_masks_match_cpu_reference_exactly(self):
        self.install()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                module = make_attention(device="cuda:0", dtype=dtype)
                expected = ORIGINAL_MASK(module, 32, 48, dtype=dtype)
                actual = self.mask(module, dtype=dtype)
                self.assertEqual(actual[0].device.type, "cuda")
                for result, reference in zip(actual, expected):
                    self.assertTrue(torch.equal(result.cpu(), reference))


if __name__ == "__main__":
    unittest.main()
