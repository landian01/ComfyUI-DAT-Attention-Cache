# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 landian01
# DAT methods adapted from Spandrel 0.4.2; see NOTICE and licenses/.
"""DAT-local copies of four methods, changing only explicit contiguous copies."""

from __future__ import annotations

import io
from functools import wraps
import hashlib
import inspect
import logging
import textwrap
import tokenize

import torch
import torch.nn.functional as F
from spandrel.architectures.DAT.__arch.DAT import img2windows

from .layout_backend import LayoutBackend


LOGGER = logging.getLogger(__name__)
BACKEND = LayoutBackend()
_contiguous = BACKEND.contiguous
MARKER = "_comfy_dat_layout_patch"


def spatial_gate_forward(self, x, H, W):
    x1, x2 = x.chunk(2, dim=-1)
    B, _N, C = x.shape
    x2 = _contiguous(
        self.conv(
            _contiguous(self.norm(x2).transpose(1, 2)).view(B, C // 2, H, W)
        ).flatten(2).transpose(-1, -2)
    )
    return x1 * x2


def spatial_attention_im2win(self, x, H, W):
    B, _N, C = x.shape
    x = _contiguous(x.transpose(-2, -1)).view(B, C, H, W)
    x = img2windows(x, self.H_sp, self.W_sp)
    x = _contiguous(
        x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
        .permute(0, 2, 1, 3)
    )
    return x


def adaptive_spatial_forward(self, x, H, W):
    """
    Input: x: (B, H*W, C), H, W
    Output: x: (B, H*W, C)
    """
    B, L, C = x.shape
    assert L == H * W, 'flatten img_tokens has wrong size'
    qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
    v = _contiguous(qkv[2].transpose(-2, -1)).view(B, C, H, W)
    max_split_size = max(self.split_size[0], self.split_size[1])
    pad_l = pad_t = 0
    pad_r = (max_split_size - W % max_split_size) % max_split_size
    pad_b = (max_split_size - H % max_split_size) % max_split_size
    qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)
    qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1)
    _H = pad_b + H
    _W = pad_r + W
    _L = _H * _W
    if (
        self.rg_idx % 2 == 0 and self.b_idx > 0 and ((self.b_idx - 2) % 4 == 0)
    ) or (self.rg_idx % 2 != 0 and self.b_idx % 4 == 0):
        qkv = qkv.view(3, B, _H, _W, C)
        qkv_0 = torch.roll(
            qkv[:, :, :, :, :C // 2],
            shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3),
        )
        qkv_0 = qkv_0.view(3, B, _L, C // 2)
        qkv_1 = torch.roll(
            qkv[:, :, :, :, C // 2:],
            shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3),
        )
        qkv_1 = qkv_1.view(3, B, _L, C // 2)
        if self.patches_resolution != _H or self.patches_resolution != _W:
            mask_tmp = self.calculate_mask(_H, _W, dtype=x.dtype)
            x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
            x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
        else:
            x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
            x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)
        x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
        x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
        x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
        attened_x = torch.cat([x1, x2], dim=2)
    else:
        x1 = self.attns[0](qkv[:, :, :, :C // 2], _H, _W)[:, :H, :W, :].reshape(
            B, L, C // 2
        )
        x2 = self.attns[1](qkv[:, :, :, C // 2:], _H, _W)[:, :H, :W, :].reshape(
            B, L, C // 2
        )
        attened_x = torch.cat([x1, x2], dim=2)
    conv_x = self.dwconv(v)
    channel_map = _contiguous(self.channel_interaction(conv_x).permute(0, 2, 3, 1)).view(B, 1, C)
    attention_reshape = _contiguous(attened_x.transpose(-2, -1)).view(B, C, H, W)
    spatial_map = self.spatial_interaction(attention_reshape)
    attened_x = attened_x * torch.sigmoid(channel_map)
    conv_x = torch.sigmoid(spatial_map) * conv_x
    conv_x = _contiguous(conv_x.permute(0, 2, 3, 1)).view(B, L, C)
    x = attened_x + conv_x
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def adaptive_channel_forward(self, x, H, W):
    """
    Input: x: (B, H*W, C), H, W
    Output: x: (B, H*W, C)
    """
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = (qkv[0], qkv[1], qkv[2])
    q = q.transpose(-2, -1)
    k = k.transpose(-2, -1)
    v = v.transpose(-2, -1)
    v_ = _contiguous(v.reshape(B, C, N)).view(B, C, H, W)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    attn = q @ k.transpose(-2, -1) * self.temperature
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
    conv_x = self.dwconv(v_)
    attention_reshape = _contiguous(attened_x.transpose(-2, -1)).view(B, C, H, W)
    channel_map = self.channel_interaction(attention_reshape)
    spatial_map = _contiguous(self.spatial_interaction(conv_x).permute(0, 2, 3, 1)).view(B, N, 1)
    attened_x = attened_x * torch.sigmoid(spatial_map)
    conv_x = conv_x * torch.sigmoid(channel_map)
    conv_x = _contiguous(conv_x.permute(0, 2, 3, 1)).view(B, N, C)
    x = attened_x + conv_x
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


METHODS = (
    ("SpatialGate", "forward", spatial_gate_forward, "7510506368d4bcb29772706c07a2d7680a2559908efc36aa7450fb99856f18b3"),
    ("Spatial_Attention", "im2win", spatial_attention_im2win, "d876f5d26858589de6da93efe405d9b9332f9d86ec20fe546b440dda8f64f85e"),
    ("Adaptive_Spatial_Attention", "forward", adaptive_spatial_forward, "9ac5eb3087229919052ca2ef18360421d2e47a533830e0aba1465bed65338ad2"),
    ("Adaptive_Channel_Attention", "forward", adaptive_channel_forward, "11f85a3b69894056a0d3f7422f0eca652616756ae4364285715ee3ae51d5e4e3"),
)


def _source_hash(method) -> str:
    # Inspect the actual implementation, without following another patch's __wrapped__.
    source = textwrap.dedent(inspect.getsource(method.__code__))
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    signature = repr([(tokenize.tok_name[token.type], token.string) for token in tokens if token.type not in ignored])
    return hashlib.sha256(signature.encode()).hexdigest()


def _wrap(original, replacement):
    @wraps(original)
    def dat_layout_method(self, x, H, W):
        if not BACKEND.enabled or self.training or torch.is_grad_enabled() or x.device.type != "cuda":
            return original(self, x, H, W)
        return replacement(self, x, H, W)

    return dat_layout_method


def install_layout_patch(dat_module=None) -> LayoutBackend | None:
    if dat_module is None:
        from spandrel.architectures.DAT.__arch import DAT as dat_module

    installed = getattr(dat_module, MARKER, None)
    if installed is not None:
        return installed

    pending = []
    try:
        for class_name, method_name, replacement, expected in METHODS:
            owner = getattr(dat_module, class_name)
            original = getattr(owner, method_name)
            if _source_hash(original) != expected:
                LOGGER.warning("DAT layout patch skipped: unsupported %s.%s source.", class_name, method_name)
                return None
            pending.append((owner, method_name, _wrap(original, replacement)))
    except (AttributeError, OSError, TypeError, SyntaxError):
        LOGGER.warning("DAT layout patch skipped: cannot verify Spandrel methods.", exc_info=True)
        return None

    for owner, name, method in pending:
        setattr(owner, name, method)
    setattr(dat_module, MARKER, BACKEND)
    return BACKEND
