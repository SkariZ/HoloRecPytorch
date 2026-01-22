# code/tracking/frame_provider.py
import numpy as np
import torch

from backend import fft_loader as fl
from backend.other_utils import create_circular_mask, create_ellipse_mask


class FFTDecoder:
    """
    Cached FFT-vector -> complex field decoder.

    Important: fft_loader.vec_to_field() mutates the mask tensor (fills vec into mask),
    so we keep a template mask and clone it for each decode.
    """
    def __init__(self, shape, pupil_radius, mask_shape="ellipse", device=None):
        self.shape = tuple(shape)  # (H,W)
        self.pupil_radius = pupil_radius
        self.mask_shape = mask_shape
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        h, w = self.shape
        if mask_shape == "circle":
            m = create_circular_mask(h, w, radius=pupil_radius)
        elif mask_shape == "ellipse":
            m = create_ellipse_mask(h, w, percent=pupil_radius / h)
        else:
            raise ValueError(f"Unknown mask_shape: {mask_shape}")

        # vec_to_field expects complex mask with 1s in pupil region
        self.mask_template = torch.tensor(m, dtype=torch.complex64, device=self.device)

    def decode(self, vec_1d):
        if not torch.is_tensor(vec_1d):
            vec_1d = torch.tensor(vec_1d, dtype=torch.complex64, device=self.device)
        else:
            vec_1d = vec_1d.to(self.device)

        field = fl.vec_to_field(
            vec_1d,
            pupil_radius=self.pupil_radius,
            shape=self.shape,
            mask=self.mask_template.clone(),   # critical (vec_to_field mutates mask)
            mask_shape=self.mask_shape,
            to_real=False,
        )
        return field  # torch complex64 (H,W)


class FrameProvider:
    """
    Provides complex frames as numpy (H,W) complex64
    for both:
      - field.npy shaped (T,H,W) (already complex field)
      - field.npy shaped (T,N)   (FFT-compressed vector)
    """
    def __init__(self, full_data_memmap: np.ndarray, fft_enabled: bool,
                 orig_size=None, pupil_radius=None, mask_shape="ellipse"):
        self.data = full_data_memmap
        self.fft_enabled = bool(fft_enabled)
        self.orig_size = tuple(orig_size) if orig_size is not None else None
        self.pupil_radius = pupil_radius
        self.mask_shape = mask_shape

        self.decoder = None
        if self.fft_enabled and self.orig_size is not None and self.pupil_radius is not None:
            self.decoder = FFTDecoder(self.orig_size, self.pupil_radius, mask_shape=self.mask_shape)

    def __len__(self):
        return int(self.data.shape[0])

    def get_complex_frame_np(self, idx: int) -> np.ndarray:
        frame = self.data[idx]

        # Non-FFT: already image-like
        if frame.ndim == 2:
            return frame.astype(np.complex64)

        # FFT-compressed vector
        if frame.ndim == 1:
            if not self.fft_enabled:
                raise ValueError("Loaded data is 1D (FFT-compressed). Enable FFT decode.")
            if self.decoder is None:
                raise ValueError("FFT decode requires orig_size (H,W) and pupil_radius.")
            field_t = self.decoder.decode(frame)
            return field_t.detach().cpu().numpy().astype(np.complex64)

        raise ValueError(f"Unexpected frame ndim={frame.ndim}, shape={frame.shape}")
