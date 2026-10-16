# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 landian01
"""FP32 transpose copies using the cuBLAS library bundled with PyTorch."""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
import sys
import threading

import torch


LOGGER = logging.getLogger(__name__)
MIN_ELEMENTS = 65536
INT_MAX = 2**31 - 1


def eligible(tensor: torch.Tensor) -> bool:
    return (
        tensor.device.type == "cuda"
        and tensor.dtype == torch.float32
        and tensor.layout == torch.strided
        and tensor.ndim in (2, 3)
        and (tensor.ndim == 2 or tensor.shape[0] == 1)
        and MIN_ELEMENTS <= tensor.numel()
        and 0 < tensor.shape[-1] <= INT_MAX
        and 0 < tensor.shape[-2] <= INT_MAX
        and tensor.stride(-2) == 1
        and tensor.shape[-2] <= tensor.stride(-1) <= INT_MAX
        and not tensor.is_contiguous()
        and not tensor.is_neg()
        and not tensor.is_conj()
    )


class CublasTranspose:
    def __init__(self, library: Path):
        self.library = ctypes.WinDLL(str(library))
        pointer = ctypes.c_void_p
        scalar = ctypes.POINTER(ctypes.c_float)
        self.get_stream = self.library.cublasGetStream_v2
        self.get_stream.argtypes = [pointer, ctypes.POINTER(pointer)]
        self.get_stream.restype = ctypes.c_int
        self.get_pointer_mode = self.library.cublasGetPointerMode_v2
        self.get_pointer_mode.argtypes = [pointer, ctypes.POINTER(ctypes.c_int)]
        self.get_pointer_mode.restype = ctypes.c_int
        self.geam = self.library.cublasSgeam
        self.geam.argtypes = [
            pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            scalar, pointer, ctypes.c_int, scalar, pointer, ctypes.c_int,
            pointer, ctypes.c_int,
        ]
        self.geam.restype = ctypes.c_int
        self.one = ctypes.c_float(1)
        self.zero = ctypes.c_float(0)

    @staticmethod
    def check(status: int) -> None:
        if status:
            raise RuntimeError(f"DAT cuBLAS transpose failed with cuBLAS status {status}")

    def copy(self, tensor: torch.Tensor) -> torch.Tensor | None:
        # PyTorch owns this per-thread/device handle and its stream/workspace.
        # Do not change its stream, pointer mode, math mode, or lifetime.
        with torch.cuda.device(tensor.device):
            if torch.cuda.is_current_stream_capturing():
                return None
            handle = ctypes.c_void_p(torch.cuda.current_blas_handle())
            stream = ctypes.c_void_p()
            pointer_mode = ctypes.c_int()
            self.check(self.get_stream(handle, ctypes.byref(stream)))
            self.check(self.get_pointer_mode(handle, ctypes.byref(pointer_mode)))
            current_stream = torch.cuda.current_stream(tensor.device).cuda_stream
            if (stream.value or 0) != current_stream or pointer_mode.value != 0:
                return None
            result = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            source = ctypes.c_void_p(tensor.data_ptr())
            self.check(self.geam(
                handle, 1, 1, tensor.shape[-1], tensor.shape[-2],
                ctypes.byref(self.one), source, tensor.stride(-1),
                ctypes.byref(self.zero), source, tensor.stride(-1),
                ctypes.c_void_p(result.data_ptr()), tensor.shape[-1],
            ))
            return result


class LayoutBackend:
    def __init__(self):
        self.enabled = True
        self._attempted = False
        self._backend: CublasTranspose | None = None
        self._lock = threading.Lock()

    def _load(self) -> CublasTranspose | None:
        with self._lock:
            if self._attempted:
                return self._backend
            self._attempted = True
            if sys.platform != "win32" or not torch.version.cuda:
                LOGGER.info("DAT layout transpose uses PyTorch on this platform.")
                return None
            major = torch.version.cuda.split(".")[0]
            library = Path(torch.__file__).resolve().parent / "lib" / f"cublas64_{major}.dll"
            try:
                self._backend = CublasTranspose(library)
            except (OSError, AttributeError):
                LOGGER.warning("DAT cuBLAS layout library unavailable; using PyTorch.", exc_info=True)
                return None
            LOGGER.info("DAT FP32 layout transpose enabled using PyTorch's bundled cuBLAS.")
            return self._backend

    def contiguous(self, tensor: torch.Tensor) -> torch.Tensor:
        if (
            not self.enabled
            or os.environ.get("DAT_CACHE_LAYOUT") == "0"
            or torch.is_grad_enabled()
            or not eligible(tensor)
            or torch.compiler.is_compiling()
            or torch.jit.is_tracing()
        ):
            return tensor.contiguous()
        backend = self._load()
        if backend is not None:
            result = backend.copy(tensor)
            if result is not None:
                return result
        return tensor.contiguous()
