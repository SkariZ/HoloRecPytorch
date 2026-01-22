# backend/tracking/preprocess.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class PreprocessConfig:
    # Spatial ops
    gaussian_subtract_sigma: float = 0.0   # 0 disables: img - G(img, sigma)
    gaussian_filter_sigma: float = 0.0     # 0 disables: G(img, sigma)
    normalize_01: bool = True

    # Normalization method
    norm_mode: str = "percentile"          # "percentile" or "minmax"
    norm_p_low: float = 1.0
    norm_p_high: float = 99.0
    eps: float = 1e-8

    # Temporal subtraction (window pairing)
    temporal_subsize: int = 0              # 0 disables
    temporal_mode: str = "forward"         # "forward" only for now (t vs t+subsize)


def normalize01(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    x = img.astype(np.float32, copy=False)

    if cfg.norm_mode == "minmax":
        lo = float(np.min(x))
        hi = float(np.max(x))
    else:
        lo, hi = np.percentile(x, (cfg.norm_p_low, cfg.norm_p_high))
        lo, hi = float(lo), float(hi)

    if hi - lo < cfg.eps:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def spatial_preprocess(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    img: 2D float/real image (already channel-extracted).
    """
    x = img.astype(np.float32, copy=False)

    # Gaussian subtraction (background/lowpass removal)
    if cfg.gaussian_subtract_sigma and cfg.gaussian_subtract_sigma > 0:
        bg = gaussian_filter(x, cfg.gaussian_subtract_sigma)
        x = x - bg

    # Gaussian smoothing
    if cfg.gaussian_filter_sigma and cfg.gaussian_filter_sigma > 0:
        x = gaussian_filter(x, cfg.gaussian_filter_sigma)

    # Normalize
    if cfg.normalize_01:
        x = normalize01(x, cfg)

    return x


def temporal_subtract_pair(
    img_t: np.ndarray,
    img_tp: np.ndarray,
    cfg: PreprocessConfig
) -> np.ndarray:
    """
    Implements your rule: frame t minus frame (t + subsize).
    Assumes img_t and img_tp are 2D float images (channel extracted).
    """
    x = img_t.astype(np.float32, copy=False) - img_tp.astype(np.float32, copy=False)

    # After temporal subtraction, you often want to normalize again
    if cfg.normalize_01:
        x = normalize01(x, cfg)
    return x
