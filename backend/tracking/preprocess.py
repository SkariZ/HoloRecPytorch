# backend/tracking/preprocess.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

from skimage.transform import resize


@dataclass
class PreprocessConfig:
    # Spatial ops
    gaussian_subtract_sigma: float = 0.0   # 0 disables: img - G(img, sigma)
    gaussian_filter_sigma: float = 0.0     # 0 disables: G(img, sigma)
    normalize_01: bool = False
    downsample_factor: int = 1   # 1=off, 2,4,...


    # Temporal subtraction (window pairing)
    temporal_subsize: int = 0


def normalize01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def _downsample(x: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample by integer factor using skimage resize (area-like for downsampling).
    """
    if factor <= 1:
        return x
    H, W = x.shape
    newH = max(1, H // factor)
    newW = max(1, W // factor)

    # preserve_range=True keeps values; anti_aliasing=True helps downsampling
    y = resize(
        x,
        (newH, newW),
        order=1,                # bilinear
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32, copy=False)
    return y

def spatial_preprocess(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    img: 2D float/real image (already channel-extracted).
    """
    x = img.astype(np.float32, copy=False)

    # Gaussian subtraction (background/lowpass removal)
    if cfg.gaussian_subtract_sigma and cfg.gaussian_subtract_sigma > 0:
        bg = gaussian_filter(x, float(cfg.gaussian_subtract_sigma))
        x = x - bg

    # Gaussian smoothing
    if cfg.gaussian_filter_sigma and cfg.gaussian_filter_sigma > 0:
        x = gaussian_filter(x, float(cfg.gaussian_filter_sigma))

    # Normalize
    if cfg.normalize_01:
        x = normalize01(x)

    # Downsample LAST
    if cfg.downsample_factor and cfg.downsample_factor > 1:
        x = _downsample(x, int(cfg.downsample_factor))

    return x


def temporal_subtract_pair(img_t: np.ndarray, img_tp: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    frame t minus frame (t + subsize).
    """
    x = img_t.astype(np.float32, copy=False) - img_tp.astype(np.float32, copy=False)

    # Normalize after subtraction
    if cfg.normalize_01:
        x = normalize01(x)

    # Downsample
    if cfg.downsample_factor and cfg.downsample_factor > 1:
        x = _downsample(x, int(cfg.downsample_factor))

    return x
