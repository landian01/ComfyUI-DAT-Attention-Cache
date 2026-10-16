import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dat_cache_native_backend_test", ROOT / "native_backend.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class NativeBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "csrc").mkdir()
        (self.root / "csrc/dat_mask_cuda.cu").write_text("test kernel", encoding="utf-8")
        self.backend = native.NativeBackend(self.root)
        self.backend.enabled = True
        self.device = torch.device("cuda:0")
        self.module = SimpleNamespace(split_size=(8, 16), shift_size=(4, 8))
        self.cap = patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)).start()
        self.addCleanup(patch.stopall)
        self.op = Mock(return_value=("first", "second"))
        patch.object(torch.ops, "dat_attention_cache_native", SimpleNamespace(make_masks=self.op)).start()
        self.load = patch.object(torch.ops, "load_library").start()

    def write_manifest(self, **changes):
        record = {"identity": native.build_identity(self.root), "compute_capability": [12, 0],
                  "library": "test.dll"}
        record.update(changes)
        (self.root / "_native").mkdir(exist_ok=True)
        (self.root / "_native/manifest.json").write_text(json.dumps(record), encoding="utf-8")

    def call(self, **changes):
        values = dict(module=self.module, height=32, width=48, device=self.device, dtype=torch.float32)
        values.update(changes)
        return self.backend.make_masks(**values)

    def test_no_build_uses_fallback_without_cuda_initialization(self):
        self.assertIsNone(self.call())
        self.assertIsNone(self.call())
        self.load.assert_not_called()
        self.cap.assert_not_called()

    def test_disabled_and_cpu_paths_do_not_load_library(self):
        self.write_manifest()
        self.backend.enabled = False
        self.assertIsNone(self.call())
        self.backend.enabled = True
        self.assertIsNone(self.call(device=torch.device("cpu")))
        self.load.assert_not_called()
        self.cap.assert_not_called()

    def test_changed_identity_and_corrupt_manifest_fall_back(self):
        for identity in ({"torch": "wrong"}, None):
            with self.subTest(identity=identity):
                self.backend = native.NativeBackend(self.root)
                self.backend.enabled = True
                self.write_manifest(identity=identity)
                with self.assertLogs(native.LOGGER, level="WARNING"):
                    self.assertIsNone(self.call())
        self.load.assert_not_called()

    def test_source_change_requires_rebuild(self):
        self.write_manifest()
        (self.root / "csrc/dat_mask_cuda.cu").write_text("new kernel", encoding="utf-8")
        with self.assertLogs(native.LOGGER, level="WARNING"):
            self.assertIsNone(self.call())
        self.load.assert_not_called()

    def test_other_gpu_falls_back_and_matching_gpu_can_load_later(self):
        self.write_manifest()
        self.cap.return_value = (7, 5)
        self.assertIsNone(self.call())
        self.load.assert_not_called()
        self.cap.return_value = (12, 0)
        self.assertEqual(self.call(), ("first", "second"))
        self.load.assert_called_once()
        self.cap.return_value = (7, 5)
        self.assertIsNone(self.call())
        self.op.assert_called_once()

    def test_compatible_library_loads_once_and_preserves_arguments(self):
        self.write_manifest()
        self.assertEqual(self.call(), ("first", "second"))
        self.assertEqual(self.call(), ("first", "second"))
        self.load.assert_called_once_with(str(self.root / "_native/test.dll"))
        self.op.assert_called_with(32, 48, 8, 16, 4, 8, self.device, torch.float32)

    def test_load_failure_warns_once_and_falls_back(self):
        self.write_manifest()
        self.load.side_effect = OSError("test missing dependency")
        with self.assertLogs(native.LOGGER, level="WARNING"):
            self.assertIsNone(self.call())
        self.assertIsNone(self.call())
        self.load.assert_called_once()

    def test_unsupported_geometry_and_dtype_use_fallback(self):
        self.write_manifest()
        for changes in ({"height": 31}, {"height": 0}, {"height": 2**32}, {"dtype": torch.float64},
                        {"module": SimpleNamespace(split_size=(64, 64), shift_size=(32, 32))},
                        {"module": SimpleNamespace(split_size=(8, 16), shift_size=(9, 8))}):
            with self.subTest(changes=changes):
                self.assertIsNone(self.call(**changes))
        self.load.assert_not_called()

    def test_runtime_error_is_propagated(self):
        self.write_manifest()
        self.op.side_effect = RuntimeError("test runtime CUDA failure")
        with self.assertRaisesRegex(RuntimeError, "runtime CUDA failure"):
            self.call()


if __name__ == "__main__":
    unittest.main()
