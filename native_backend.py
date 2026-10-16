# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 landian01
"""Optional, explicitly built LibTorch operator; never compiles during startup."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import threading

import torch

LOGGER = logging.getLogger(__name__)
INT_MAX = 2**31 - 1


def build_identity(root: Path) -> dict:
    return {
        "source_sha256": hashlib.sha256((root / "csrc/dat_mask_cuda.cu").read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "system": platform.system(),
        "machine": platform.machine(),
        "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    }


class NativeBackend:
    def __init__(self, root: Path):
        self.root = root
        self.enabled = os.environ.get("DAT_CACHE_NATIVE", "1") != "0"
        self._attempted = False
        self._op = None
        self._capability = None
        self._library = None
        self._lock = threading.Lock()

    def available(self, device: torch.device) -> bool:
        if not self.enabled or device.type != "cuda":
            return False
        with self._lock:
            try:
                if not self._attempted:
                    self._attempted = True
                    manifest = self.root / "_native/manifest.json"
                    if manifest.is_file():
                        record = json.loads(manifest.read_text(encoding="utf-8"))
                        if record["identity"] != build_identity(self.root):
                            raise ValueError("PyTorch, CUDA, platform or kernel source changed; rebuild")
                        library_name = record["library"]
                        if not isinstance(library_name, str) or Path(library_name).name != library_name:
                            raise ValueError("Invalid native library filename")
                        self._capability = tuple(record["compute_capability"])
                        if len(self._capability) != 2 or not all(isinstance(v, int) and v >= 0 for v in self._capability):
                            raise ValueError("Invalid GPU compute capability")
                        self._library = manifest.parent / library_name
                if self._library is None or self._capability != torch.cuda.get_device_capability(device):
                    return False
                if self._op is None:
                    torch.ops.load_library(str(self._library))
                    self._op = torch.ops.dat_attention_cache_native.make_masks
                    LOGGER.info("DAT native CUDA mask backend enabled for SM %s%s.", *self._capability)
                return True
            except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError):
                self._library = None
                self._op = None
                LOGGER.warning("DAT native backend unavailable; using PyTorch.", exc_info=True)
                return False

    def make_masks(self, module, height: int, width: int, device: torch.device, dtype: torch.dtype):
        if not self.enabled or device.type != "cuda" or dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return None
        split_h, split_w = map(int, module.split_size)
        shift_h, shift_w = map(int, module.shift_size)
        height, width = int(height), int(width)
        if not (0 < height <= INT_MAX and 0 < width <= INT_MAX
                and split_h > 0 and split_w > 0 and split_h * split_w <= 1024
                and 0 <= shift_h <= split_h and 0 <= shift_w <= split_w):
            return None
        if height % split_h or height % split_w or width % split_h or width % split_w:
            return None
        if (height // split_h) * (width // split_w) > INT_MAX or not self.available(device):
            return None
        # Runtime CUDA failures propagate: continuing after an illegal access is unsafe.
        return self._op(height, width, split_h, split_w, shift_h, shift_w, device, dtype)
