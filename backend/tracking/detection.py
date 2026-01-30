# backend/tracking/detection.py  (additions)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Type
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.ndimage import gaussian_laplace

import numpy as np

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
    """
    All detectors must output a list of Detection(y,x,score).
    Score should be comparable within a detector.
    """
    name: str = "Base"

    def __init__(self, threshold: float = 0.0, top_k: Optional[int] = None):
        self.threshold = float(threshold)
        self.top_k = top_k

    @abstractmethod
    def detect(self, img2d: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class PeakDetector(DetectorBase):
    name = "Peaks (local max)"

    def __init__(self, threshold: float = 0.0, top_k: Optional[int] = None, win_size: int = 9):
        super().__init__(threshold=threshold, top_k=top_k)
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
        return [Detection(float(ys[i]), float(xs[i]), float(scores[i])) for i in order]


class LoGDetector(DetectorBase):
    """
    Blob-ish detector using Laplacian of Gaussian response.
    Good baseline alternative without many knobs.
    """
    name = "LoG (blobs)"

    def __init__(self, threshold: float = 0.0, top_k: Optional[int] = None, sigma: float = 2.0):
        super().__init__(threshold=threshold, top_k=top_k)
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

        return [Detection(float(ys[i]), float(xs[i]), float(scores[i])) for i in order]

class UnetBaseDetector(DetectorBase):
    """
    Base class for U-Net based detectors.
    Subclasses must implement the _predict method.
    """
    name = "U-Net Base"

    def __init__(self, threshold: float = 0.5, top_k: Optional[int] = None):
        super().__init__(threshold=threshold, top_k=top_k)
        # Load model weights here if needed

        self.model = None  # Placeholder for the U-Net model
        

    @abstractmethod
    def _predict(self, img2d: np.ndarray) -> np.ndarray:
        """
        Predict a probability map from the input image.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def detect(self, img2d: np.ndarray) -> List[Detection]:
        prob_map = self._predict(img2d)
        peaks = (prob_map >= self.threshold)

        ys, xs = np.nonzero(peaks)
        if len(xs) == 0:
            return []

        scores = prob_map[ys, xs]
        order = np.argsort(scores)[::-1]
        if self.top_k is not None and self.top_k > 0:
            order = order[: self.top_k]

        return [Detection(float(ys[i]), float(xs[i]), float(scores[i])) for i in order]


# Registry (UI uses this)
DETECTOR_REGISTRY: Dict[str, type] = {
    PeakDetector.name: PeakDetector,
    LoGDetector.name: LoGDetector,
    UnetBaseDetector.name: UnetBaseDetector,
}

PeakDetection = Detection