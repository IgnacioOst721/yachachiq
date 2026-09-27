"""Scale stability rule, identical to the firmware (firmware/box_esp32/src/scale.cpp).

A window of the last `n` samples is stable when
    sigma(window) <= stable_g   and   |mean(first half) - mean(second half)| <= stable_g
(the second test catches slow drift/creep that a small sigma alone misses).
Samples are converted to grams with the calibration factor (counts per gram).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class WindowStats:
    mean: float        # same unit as the samples
    sigma: float       # sample standard deviation (n-1)
    drift: float       # |mean(first half) - mean(second half)|
    n: int


def window_stats(samples: Sequence[float]) -> WindowStats:
    n = len(samples)
    if n < 3:
        raise ValueError("need at least 3 samples")
    m = sum(samples) / n
    var = sum((s - m) ** 2 for s in samples) / (n - 1)
    half = n // 2
    h1 = sum(samples[:half]) / half
    h2 = sum(samples[half:]) / (n - half)
    return WindowStats(mean=m, sigma=math.sqrt(var), drift=abs(h1 - h2), n=n)


def is_stable(samples_g: Sequence[float], stable_g: float) -> bool:
    st = window_stats(samples_g)
    return st.sigma <= stable_g and st.drift <= stable_g


def find_stable_window(samples_g: Sequence[float], n: int, stable_g: float) -> Optional[int]:
    """Index of the end (exclusive) of the first stable window of length n, or None.
    This is what the firmware does sample by sample."""
    for end in range(n, len(samples_g) + 1):
        if is_stable(samples_g[end - n:end], stable_g):
            return end
    return None


def raw_to_grams(raw: float, offset: float, factor: float) -> float:
    return (raw - offset) / factor
