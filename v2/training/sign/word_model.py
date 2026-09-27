"""Skeleton transformer for isolated signs (small enough for the Jetson, exports to ONNX).

Input (B, T, F) normalised keypoint features (features.normalize_sequence, T=32 frames).
  1. per-frame MLP embedding (F -> D)
  2. two depthwise temporal convolutions (local motion, k=5) with residuals
  3. N transformer encoder blocks (hand-written, so ONNX export is plain MatMul/Softmax)
  4. mean + max pooling over time -> classifier
Frames that are all zero (no person) are masked out of attention and pooling.
Same family as the top Kaggle ISLR 2023 solutions (1D-CNN + transformer on landmarks).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class ConvBlock(nn.Module):
    def __init__(self, d: int, k: int = 5, drop: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.pw = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h = self.dw(h).transpose(1, 2)
        return x + self.drop(self.pw(Fn.silu(h)))


class Block(nn.Module):
    def __init__(self, d: int, heads: int = 4, ff: int = 2, drop: float = 0.1):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Dropout(drop), nn.Linear(ff * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, bias):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        hd = d // self.h
        q = q.reshape(b, t, self.h, hd).transpose(1, 2)
        k = k.reshape(b, t, self.h, hd).transpose(1, 2)
        v = v.reshape(b, t, self.h, hd).transpose(1, 2)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd) + bias      # bias: (b,1,1,t)
        att = torch.softmax(att, dim=-1)
        h = torch.matmul(att, v).transpose(1, 2).reshape(b, t, d)
        x = x + self.drop(self.proj(h))
        return x + self.drop(self.ff(self.n2(x)))


class SignTransformer(nn.Module):
    def __init__(self, n_feat: int, n_cls: int, frames: int = 32, d: int = 192, layers: int = 3,
                 heads: int = 4, drop: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_feat, d), nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, frames, d))
        self.convs = nn.ModuleList([ConvBlock(d, drop=drop) for _ in range(2)])
        self.blocks = nn.ModuleList([Block(d, heads, drop=drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * d, n_cls))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        valid = (x.abs().sum(-1) > 0).float()                        # (b, t)
        bias = ((1.0 - valid) * -1e4)[:, None, None, :]
        h = self.embed(x) + self.pos
        for c in self.convs:
            h = c(h)
        for blk in self.blocks:
            h = blk(h, bias)
        h = self.norm(h)
        w = valid[..., None]
        mean = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        mx = (h + (w - 1.0) * 1e4).max(1).values
        return self.head(torch.cat([mean, mx], dim=-1))
