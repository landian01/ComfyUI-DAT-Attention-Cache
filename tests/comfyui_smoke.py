import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(os.environ.get("COMFYUI_ROOT"), "Set COMFYUI_ROOT for the real ComfyUI integration test")
class ComfyUIIntegrationTests(unittest.TestCase):
    def test_real_upscale_node_cleans_cache_on_success_and_failure(self):
        sys.path.insert(0, os.environ["COMFYUI_ROOT"])
        import comfy.cli_args

        comfy.cli_args.args.cpu = True
        from comfy import model_management
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        from spandrel.architectures.DAT.__arch.DAT import Adaptive_Spatial_Attention as Attention

        missing = object()
        attention_names = ("calculate_mask", "clear_runtime_mask_cache", "_comfy_dat_mask_cache_patch")
        upscale_names = ("execute", "upscale", "_comfy_dat_cache_cleanup_patch")
        saved = [(Attention, name, Attention.__dict__.get(name, missing)) for name in attention_names]
        saved += [(ImageUpscaleWithModel, name, ImageUpscaleWithModel.__dict__.get(name, missing)) for name in upscale_names]
        spec = importlib.util.spec_from_file_location("dat_cache_real_core_test", Path(__file__).resolve().parents[1] / "__init__.py")
        plugin = importlib.util.module_from_spec(spec)
        try:
            with patch.dict(sys.modules, {spec.name: plugin}):
                spec.loader.exec_module(plugin)
            self.assertTrue(plugin.PATCH_ACTIVE)
            attention = torch.nn.Module()
            attention.register_parameter("weight", torch.nn.Parameter(torch.empty(1)))
            attention.split_size = (8, 16)
            attention.shift_size = (4, 8)

            class Model:
                scale = 1
                patcher = SimpleNamespace(load_device=torch.device("cpu"))
                fail = False

                def __call__(self, image):
                    Attention.calculate_mask(attention, 32, 32, dtype=image.dtype)
                    if self.fail:
                        raise RuntimeError("expected integration failure")
                    return image * 0.5

            model = Model()
            image = torch.rand((1, 16, 16, 3))
            with patch.object(model_management, "load_models_gpu"):
                result, = ImageUpscaleWithModel.execute(model, image)
                torch.testing.assert_close(result, image * 0.5)
                self.assertFalse(plugin._runtime_mask_cache)
                model.fail = True
                with self.assertRaisesRegex(RuntimeError, "expected integration failure"):
                    ImageUpscaleWithModel.execute(model, image)
                self.assertFalse(plugin._runtime_mask_cache)
                self.assertFalse(plugin._runtime_shape)
        finally:
            plugin.clear_runtime_mask_cache()
            for owner, name, value in saved:
                if value is missing:
                    if name in owner.__dict__:
                        delattr(owner, name)
                else:
                    setattr(owner, name, value)


if __name__ == "__main__":
    unittest.main()
