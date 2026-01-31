# backend/tracking/detection.py  (additions)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Type, Any
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import gaussian_laplace


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

@dataclass
class Detection:
    y: float
    x: float
    score: float   # higher = more confident

class DetectorBase(ABC):
    name: str = "Base"

    def __init__(self, threshold: float = 0.0, top_k: Optional[int] = None, border_skip: int = 0):
        self.threshold = float(threshold)
        self.top_k = top_k
        self.border_skip = int(border_skip)

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

    @abstractmethod
    def detect(self, img2d: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class PeakDetector(DetectorBase):
    name = "Peaks (local max)"

    def __init__(
        self,
        threshold: float = 0.0,
        top_k: Optional[int] = None,
        win_size: int = 9,
        border_skip: int = 0,
    ):
        super().__init__(threshold=threshold, top_k=top_k, border_skip=border_skip)
        self.win_size = int(win_size)

    def detect(self, img2d: np.ndarray) -> List[Detection]:
        sm = img2d.astype(np.float32)
        mf = maximum_filter(sm, size=self.win_size, mode="nearest")
        peaks = (sm == mf) & (sm >= self.threshold)
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            return []
        scores = sm[ys, xs]
        order = np.argsort(scores)[::-1]
        if self.top_k is not None and self.top_k > 0:
            order = order[: self.top_k]

        dets = [Detection(float(ys[i]), float(xs[i]), float(scores[i])) for i in order]
        return self._filter_border(dets, sm.shape)


class LoGDetector(DetectorBase):
    """
    Blob-ish detector using Laplacian of Gaussian response.
    Good baseline alternative without many knobs.
    """
    name = "LoG (blobs)"

    def __init__(
        self,
        threshold: float = 0.0,
        top_k: Optional[int] = None,
        sigma: float = 2.0,
        border_skip: int = 0,
    ):
        super().__init__(threshold=threshold, top_k=top_k, border_skip=border_skip)
        self.sigma = float(sigma)

    def detect(self, img2d: np.ndarray) -> List[Detection]:
        x = img2d.astype(np.float32)
        resp = np.abs(gaussian_laplace(x, sigma=self.sigma)).astype(np.float32)

        # local maxima on response
        mf = maximum_filter(resp, size=9, mode="nearest")  # you can expose size later if needed
        peaks = (resp == mf) & (resp >= self.threshold)

        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            return []

        scores = resp[ys, xs]
        order = np.argsort(scores)[::-1]
        if self.top_k is not None and self.top_k > 0:
            order = order[: self.top_k]

        dets = [Detection(float(ys[i]), float(xs[i]), float(scores[i])) for i in order]

        return self._filter_border(dets, resp.shape)



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
            logits = self.model(t)
            if logits.dim() == 3:
                logits = logits.unsqueeze(1)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        return prob[:H, :W]

    def detect(self, img2d: np.ndarray) -> List[Detection]:
        score = self._predict_score_map(img2d).astype(np.float32)

        mf = maximum_filter(score, size=9, mode="nearest")
        peaks = (score == mf) & (score >= float(self.threshold))
        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            return []

        scores = score[ys, xs]
        order = np.argsort(scores)[::-1]
        if self.top_k is not None and self.top_k > 0:
            order = order[: self.top_k]

        dets = [Detection(float(ys[i]), float(xs[i]), float(scores[i])) for i in order]
        return self._filter_border(dets, score.shape)


# Registry (UI uses this)
DETECTOR_REGISTRY: Dict[str, type] = {
    PeakDetector.name: PeakDetector,
    LoGDetector.name: LoGDetector,
    UnetDetector.name: UnetDetector,
}

PeakDetection = Detection