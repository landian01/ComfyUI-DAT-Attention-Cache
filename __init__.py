# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 landian01
# DAT/Swin mask construction is adapted from the sources listed in NOTICE.

from __future__ import annotations

import inspect
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from .native_backend import NativeBackend


LOGGER = logging.getLogger(__name__)
PATCH_VERSION = "1.2.0"
MAX_VARIANTS_PER_DEVICE = 8

_mask_backend = NativeBackend(Path(__file__).resolve().parent)
_cache_lock = threading.RLock()
_runtime_mask_cache: dict[
    tuple[str, int | None],
    OrderedDict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]],
] = {}
_runtime_shape: dict[tuple[str, int | None], tuple[int, int]] = {}
_native_clear_cache = None


def clear_runtime_mask_cache() -> None:
    with _cache_lock:
        _runtime_mask_cache.clear()
        _runtime_shape.clear()

    if _native_clear_cache is not None:
        _native_clear_cache()


def _module_device_and_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(module.parameters())
    except StopIteration:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


def _calculate_mask_on_device(self, H, W, device, dtype):
    img_mask_0 = torch.zeros((1, H, W, 1), device=device, dtype=dtype)
    img_mask_1 = torch.zeros((1, H, W, 1), device=device, dtype=dtype)
    h_slices_0 = (
        slice(0, -self.split_size[0]),
        slice(-self.split_size[0], -self.shift_size[0]),
        slice(-self.shift_size[0], None),
    )
    w_slices_0 = (
        slice(0, -self.split_size[1]),
        slice(-self.split_size[1], -self.shift_size[1]),
        slice(-self.shift_size[1], None),
    )
    h_slices_1 = (
        slice(0, -self.split_size[1]),
        slice(-self.split_size[1], -self.shift_size[1]),
        slice(-self.shift_size[1], None),
    )
    w_slices_1 = (
        slice(0, -self.split_size[0]),
        slice(-self.split_size[0], -self.shift_size[0]),
        slice(-self.shift_size[0], None),
    )

    count = 0
    for h_slice in h_slices_0:
        for w_slice in w_slices_0:
            img_mask_0[:, h_slice, w_slice, :] = count
            count += 1

    count = 0
    for h_slice in h_slices_1:
        for w_slice in w_slices_1:
            img_mask_1[:, h_slice, w_slice, :] = count
            count += 1

    img_mask_0 = img_mask_0.view(
        1,
        H // self.split_size[0],
        self.split_size[0],
        W // self.split_size[1],
        self.split_size[1],
        1,
    )
    img_mask_0 = (
        img_mask_0.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, self.split_size[0], self.split_size[1], 1)
    )
    mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
    attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
    attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, -100.0).masked_fill(
        attn_mask_0 == 0, 0.0
    )

    img_mask_1 = img_mask_1.view(
        1,
        H // self.split_size[1],
        self.split_size[1],
        W // self.split_size[0],
        self.split_size[0],
        1,
    )
    img_mask_1 = (
        img_mask_1.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, self.split_size[1], self.split_size[0], 1)
    )
    mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
    attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
    attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, -100.0).masked_fill(
        attn_mask_1 == 0, 0.0
    )

    return attn_mask_0, attn_mask_1


def _install_dat_mask_cache() -> bool:
    global _native_clear_cache

    from spandrel.architectures.DAT.__arch.DAT import Adaptive_Spatial_Attention

    if getattr(Adaptive_Spatial_Attention, "_comfy_dat_mask_cache_patch", False):
        return True

    native_get_runtime_mask = getattr(Adaptive_Spatial_Attention, "get_runtime_mask", None)
    native_clear = getattr(Adaptive_Spatial_Attention, "clear_runtime_mask_cache", None)
    if callable(native_get_runtime_mask):
        if callable(native_clear):
            _native_clear_cache = native_clear
        LOGGER.info(
            "DAT already provides a runtime attention-mask cache; using the native implementation."
        )
        Adaptive_Spatial_Attention._comfy_dat_mask_cache_patch = True
        return True

    original_calculate_mask = Adaptive_Spatial_Attention.calculate_mask
    parameters = tuple(inspect.signature(original_calculate_mask).parameters)
    if parameters[:3] != ("self", "H", "W"):
        raise RuntimeError(
            "Unsupported DAT calculate_mask signature: "
            f"{inspect.signature(original_calculate_mask)}"
        )

    def calculate_mask_cached(self, H, W, dtype=None):
        device, module_dtype = _module_device_and_dtype(self)
        dtype = module_dtype if dtype is None else dtype
        device_key = (device.type, device.index)
        shape_key = (int(H), int(W))
        cache_key = (
            shape_key,
            tuple(int(value) for value in self.split_size),
            tuple(int(value) for value in self.shift_size),
            dtype,
        )

        with _cache_lock:
            if _runtime_shape.get(device_key) != shape_key:
                _runtime_mask_cache.pop(device_key, None)
                _runtime_shape[device_key] = shape_key

            variants = _runtime_mask_cache.setdefault(device_key, OrderedDict())
            cached = variants.get(cache_key)
            if cached is not None:
                variants.move_to_end(cache_key)
                return cached

            masks = _mask_backend.make_masks(self, H, W, device, dtype)
            if masks is None:
                masks = _calculate_mask_on_device(self, H, W, device, dtype)
            variants[cache_key] = masks
            while len(variants) > MAX_VARIANTS_PER_DEVICE:
                variants.popitem(last=False)
            return masks

    calculate_mask_cached._comfy_dat_mask_cache_patch = True
    Adaptive_Spatial_Attention.calculate_mask = calculate_mask_cached
    Adaptive_Spatial_Attention.clear_runtime_mask_cache = classmethod(
        lambda cls: clear_runtime_mask_cache()
    )
    Adaptive_Spatial_Attention._comfy_dat_mask_cache_patch = True
    return True


def _install_upscale_cleanup() -> bool:
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel

    if getattr(ImageUpscaleWithModel, "_comfy_dat_cache_cleanup_patch", False):
        return True

    execute_descriptor = ImageUpscaleWithModel.__dict__.get("execute")
    if not isinstance(execute_descriptor, classmethod):
        raise RuntimeError("Unsupported ImageUpscaleWithModel.execute descriptor")

    original_execute = execute_descriptor.__func__

    def execute_with_cache_cleanup(cls, *args, **kwargs):
        try:
            return original_execute(cls, *args, **kwargs)
        finally:
            clear_runtime_mask_cache()

    wrapped_execute = classmethod(execute_with_cache_cleanup)
    ImageUpscaleWithModel.execute = wrapped_execute
    ImageUpscaleWithModel.upscale = wrapped_execute
    ImageUpscaleWithModel._comfy_dat_cache_cleanup_patch = True
    return True


def _install_patch() -> bool:
    try:
        cleanup_ready = _install_upscale_cleanup()
        dat_ready = _install_dat_mask_cache()
    except Exception:
        LOGGER.exception("Failed to install the DAT attention-mask cache patch.")
        return False

    LOGGER.info(
        "DAT attention-mask cache patch %s enabled (DAT=%s, cleanup=%s).",
        PATCH_VERSION,
        dat_ready,
        cleanup_ready,
    )
    return dat_ready and cleanup_ready


def _install_layout_optimization():
    try:
        from .dat_layout import install_layout_patch
        return install_layout_patch()
    except Exception:
        LOGGER.exception("DAT layout patch unavailable; keeping the attention-mask cache.")
        return None


PATCH_ACTIVE = _install_patch()
LAYOUT_BACKEND = _install_layout_optimization() if PATCH_ACTIVE else None
LAYOUT_ACTIVE = LAYOUT_BACKEND is not None

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "PATCH_ACTIVE", "LAYOUT_ACTIVE"]
