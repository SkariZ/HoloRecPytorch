# backend/tracking/detection.py  (additions)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Type, Any, Tuple
import math
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import gaussian_laplace

from backend.tracking.imgrvt import rvt

import torch


def channel_from_complex(field: np.ndarray, channel: str) -> np.ndarray:
    """
    Extract a real-valued 2D image from a complex field.
    """
    if channel == "real":
        return np.real(field)
    elif channel == "imag":
        return np.imag(field)
    elif channel == "abs":
        return np.abs(field)
    elif channel == "phase":
        return np.angle(field)
    else:
        raise ValueError(f"Unknown channel: {channel}")

def normalize_score_map(resp: np.ndarray, p_lo=1.0, p_hi=99.9) -> np.ndarray:
    r = resp.astype(np.float32, copy=False)
    lo = np.percentile(r, p_lo)
    hi = np.percentile(r, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(r, dtype=np.float32)
    s = (r - lo) / (hi - lo + 1e-8)
    return np.clip(s, 0.0, 1.0).astype(np.float32)

@dataclass
class Detection:
    y: float
    x: float
    score: float   # higher = more confident

class DetectorBase(ABC):
    name: str = "Base"

    def __init__(self, threshold: float = 0.0, top_k: Optional[int] = None, border_skip: int = 0, min_dist: float = 7.0):
        self.threshold = float(threshold)
        self.top_k = top_k
        self.border_skip = int(border_skip)
        self.min_dist = float(min_dist)

    def _filter_border(self, dets: List[Detection], shape_hw) -> List[Detection]:
        """
        Removes detections within border_skip pixels of the image boundary.
        shape_hw: (H, W)
        """
        b = int(self.border_skip)
        if b <= 0:
            return dets
        H, W = int(shape_hw[0]), int(shape_hw[1])
        out = []
        for d in dets:
            if (d.x >= b) and (d.x < W - b) and (d.y >= b) and (d.y < H - b):
                out.append(d)
        return out

    def _postprocess(self, dets: List[Detection], shape_hw) -> List[Detection]:
        dets = self._filter_border(dets, shape_hw)
        dets = nms_min_distance(dets, self.min_dist)
        return dets

def nms_min_distance(dets: List[Detection], min_dist: float) -> List[Detection]:
    """
    Greedy NMS with spatial hashing grid. Keeps highest-score dets and suppresses
    any within min_dist pixels of a kept detection. Fast in practice.
    """
    
    if min_dist <= 0 or len(dets) <= 1:
        return dets

    cell = float(min_dist)
    cell2 = cell * cell

    dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)

    grid: Dict[Tuple[int, int], List[Detection]] = {}
    kept: List[Detection] = []

    for d in dets_sorted:
        cx = int(math.floor(d.x / cell))
        cy = int(math.floor(d.y / cell))

        ok = True
        for ny in (cy - 1, cy, cy + 1):
            for nx in (cx - 1, cx, cx + 1):
                for k in grid.get((nx, ny), []):
                    dx = d.x - k.x
                    dy = d.y - k.y
                    if dx*dx + dy*dy < cell2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            kept.append(d)
            grid.setdefault((cx, cy), []).append(d)

    return kept


class PeakDetector(DetectorBase):
    name = "Peaks (local max)"

    def __init__(
        self,
        threshold: float = 0.0,          # now interpreted as normalized 0..1
        top_k: Optional[int] = None,
        win_size: int = 9,
        border_skip: int = 0,
    ):
        super().__init__(threshold=threshold, top_k=top_k, border_skip=border_skip)
        self.win_size = int(win_size)

    def detect(self, img2d: np.ndarray) -> List[Detection]:
        sm = img2d.astype(np.float32, copy=False)

        # Normalize to 0..1 so threshold is universal
        score = normalize_score_map(sm)  # (H,W) in [0,1]

        mf = maximum_filter(score, size=self.win_size, mode="nearest")

        eps = 1e-10
        peaks = (score == mf - eps) & (score >= float(self.threshold))

        ys, xs = np.nonzero(peaks)
        if xs.size == 0:
            return []

        scores = score[ys, xs].astype(np.float32)

        order = np.argsort(scores)[::-1]
        if self.top_k is not None and self.top_k > 0:
            order = order[: self.top_k]

        dets = [Detection(float(ys[i]), float(xs[i]), float(scores[i])) for i in order]

        return self._postprocess(dets, score.shape)


class LoGDetector(DetectorBase):
    """
    Blob-ish detector using Laplacian of Gaussian response.
    Good baseline alternative without many knobs.
    """
    name = "LoG (blobs)"

    def __init__(
        self,
        threshold: float = 0.0,          # now interpreted as normalized 0..1
        top_k: Optional[int] = None,
        sigma: float = 2.0,
        border_skip: int = 0,
        win_size: int = 9,
    ):
        super().__init__(threshold=threshold, top_k=top_k, border_skip=border_skip)
        self.sigma = float(sigma)
        self.win_size = int(win_size)

    def detect(self, img2d: np.ndarray) -> List[Detection]:
        x = img2d.astype(np.float32, copy=False)

        # raw response (unbounded)
        resp = np.abs(gaussian_laplace(x, sigma=self.sigma)).astype(np.float32)

        # normalize to 0..1 so threshold is universal
        score = normalize_score_map(resp)  # (H,W) in [0,1]

        mf = maximum_filter(score, size=self.win_size, mode="nearest")

        eps = 1e-10
        peaks = (score == mf - eps) & (score >= float(self.threshold))

        ys, xs = np.nonzero(peaks)
        if xs.size == 0:
            return []

        scores = score[ys, xs].astype(np.float32)

        order = np.argsort(scores)[::-1]
        if self.top_k is not None and self.top_k > 0:
            order = order[: self.top_k]

        dets = [Detection(float(ys[i]), float(xs[i]), float(scores[i])) for i in order]

        return self._postprocess(dets, score.shape)


class UnetDetector(DetectorBase):
    name = "U-Net"

    def __init__(
        self,
        model_spec: Dict[str, Any],
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        border_skip: int = 0,
        device: Optional[str] = None,
    ):
        super().__init__(threshold=threshold, top_k=top_k, border_skip=border_skip)
        self.spec = model_spec
        self.size_factor = int(model_spec.get("size_factor", 16))

        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = model_spec["builder"]().to(self.device)
        self._load_weights(model_spec["weights"])
        self.model.eval()

    def _load_weights(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if isinstance(state, dict) and any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)

    def _predict_score_map(self, img2d: np.ndarray) -> np.ndarray:
        x = img2d.astype(np.float32, copy=False)

        # Normalize to [0,1]
        x -= x.min()
        x /= x.max() + 1e-8
        
        H, W = x.shape
        f = self.size_factor
        pad_h = (f - H % f) % f
        pad_w = (f - W % f) % f
        if pad_h or pad_w:
            x = np.pad(x, ((0, pad_h), (0, pad_w)), mode="reflect")

        t = torch.from_numpy(x)[None, None, ...].to(self.device)  # (1,1,H,W)

        with torch.no_grad():
            pred = self.model(t)

            # Normalize output to [0,1]
            prob = normalize_score_map(pred.squeeze().cpu().numpy().astype(np.float32))

        return prob[:H, :W]

    def detect(self, img2d: np.ndarray) -> List[Detection]:
        score = self._predict_score_map(img2d).astype(np.float32)

        # Smooth score map a bit to avoid noisy peaks
        score = gaussian_filter(score, sigma=1.0)
        mf = maximum_filter(score, size=9, mode="nearest")

        eps = 1e-10
        peaks = (score == mf - eps) & (score >= float(self.threshold))

        ys, xs = np.nonzero(peaks)
        if xs.size == 0:
            return []

        scores = score[ys, xs]
        order = np.argsort(scores)[::-1]
        if self.top_k is not None and self.top_k > 0:
            order = order[: self.top_k]

        dets = [Detection(float(ys[i]), float(xs[i]), float(scores[i])) for i in order]

        return self._postprocess(dets, score.shape)

class RVTDetector(DetectorBase):
    name = "RVT (radial variance)"

    def __init__(
        self,
        rmin: int = 2,
        rmax: int = 6,
        kind: str = "basic",
        highpass_size: float | None = None,
        upsample: int = 1,
        coarse_factor: int = 1,
        coarse_mode: str = "add",
        pad_mode: str = "reflect",
        peak_win: int = 9,
        threshold: float = 0.25,          # interpreted as normalized 0..1
        top_k: int | None = None,
        border_skip: int = 16,
    ):
        super().__init__(threshold=threshold, top_k=top_k, border_skip=border_skip)
        self.rmin = int(rmin)
        self.rmax = int(rmax)
        self.kind = str(kind)
        self.highpass_size = highpass_size if highpass_size is None else float(highpass_size)
        self.upsample = int(upsample)
        self.coarse_factor = int(coarse_factor)
        self.coarse_mode = str(coarse_mode)
        self.pad_mode = str(pad_mode)
        self.peak_win = int(peak_win)

    def detect(self, img2d: np.ndarray) -> List[Detection]:
        img = np.asarray(img2d, dtype=np.float32)

        # raw RVT response (unbounded, >=0)
        resp = rvt(
            img,
            rmin=self.rmin,
            rmax=self.rmax,
            kind=self.kind,
            highpass_size=self.highpass_size,
            upsample=self.upsample,
            rweights=None,
            coarse_factor=self.coarse_factor,
            coarse_mode=self.coarse_mode,
            pad_mode=self.pad_mode,
        ).astype(np.float32)

        # universal normalized score in [0,1]
        score = normalize_score_map(resp)

        win = max(3, int(self.peak_win))
        if win % 2 == 0:
            win += 1

        mf = maximum_filter(score, size=win, mode="nearest")
        peaks = (score == mf) & (score >= float(self.threshold))

        ys, xs = np.nonzero(peaks)
        if xs.size == 0:
            return []

        scores = score[ys, xs].astype(np.float32)

        order = np.argsort(scores)[::-1]
        if self.top_k is not None and self.top_k > 0:
            order = order[: self.top_k]

        xs = xs[order].astype(np.float32)
        ys = ys[order].astype(np.float32)
        scores = scores[order].astype(np.float32)

        # Map coordinates back if RVT did upsampling
        if self.upsample > 1:
            xs = (xs + 0.5) / self.upsample - 0.5
            ys = (ys + 0.5) / self.upsample - 0.5

        dets = [Detection(float(y), float(x), float(s)) for x, y, s in zip(xs, ys, scores)]

        return self._postprocess(dets, score.shape)

# Registry (UI uses this)
DETECTOR_REGISTRY: Dict[str, type] = {
    PeakDetector.name: PeakDetector,
    LoGDetector.name: LoGDetector,
    UnetDetector.name: UnetDetector,
    RVTDetector.name: RVTDetector,
}

PeakDetection = Detection