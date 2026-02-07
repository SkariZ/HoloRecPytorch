import numpy as np
from scipy.optimize import curve_fit

def compute_patch_metrics(patch: np.ndarray) -> dict:
    amp = np.abs(patch).astype(np.float32)
    ph = np.unwrap(np.angle(patch)).astype(np.float32)
    re = np.real(patch).astype(np.float32)
    im = np.imag(patch).astype(np.float32)

    return {
        "amp_mean": float(amp.mean()),
        "amp_std": float(amp.std()),
        "amp_max": float(amp.max()),
        "ph_mean": float(ph.mean()),
        "ph_std": float(ph.std()),
        "re_mean": float(re.mean()),
        "im_mean": float(im.mean()),
    }


def gaussian_mass_abs2(patch_complex):
    I = np.abs(patch_complex) ** 2
    I = I.astype(np.float32)

    h, w = I.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx - (w - 1) / 2).ravel()
    y = (yy - (h - 1) / 2).ravel()
    z = I.ravel()

    def model(X, a, x0, y0, s, c):
        x, y = X
        r2 = (x - x0) ** 2 + (y - y0) ** 2
        return a * np.exp(-r2 / (2 * s ** 2)) + c

    c0 = np.median(z)
    a0 = float(np.max(z) - c0)
    s0 = max(0.8, min(h, w) / 6)

    a_lo, a_hi = 0.0, max(1e-6, float(np.max(z) - np.min(z)) * 10)
    s_lo, s_hi = 0.5, max(1.0, min(h, w))
    c_lo, c_hi = float(np.min(z)), float(np.max(z))

    p0 = [a0, 0.0, 0.0, s0, c0]
    bounds = ([a_lo, -w / 2, -h / 2, s_lo, c_lo],
              [a_hi,  w / 2,  h / 2, s_hi, c_hi])

    popt, _ = curve_fit(model, (x, y), z, p0=p0, bounds=bounds, maxfev=5000)
    a, x0, y0, s, c = popt

    # continuous 2D Gaussian integral
    return float(2 * np.pi * a * s ** 2)


def track_motion_metrics(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    n = int(len(x))
    if n <= 1:
        return {
            "n_frames": n,
            "total_distance_px": 0.0,
            "net_displacement_px": 0.0,
            "avg_speed_px_per_frame": 0.0,
            "max_step_px": 0.0,
        }

    dx = np.diff(x.astype(np.float32))
    dy = np.diff(y.astype(np.float32))
    steps = np.sqrt(dx * dx + dy * dy)

    total_dist = float(np.sum(steps))
    net_disp = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    avg_speed = float(total_dist / max(1, n - 1))
    max_step = float(np.max(steps)) if steps.size else 0.0

    return {
        "n_frames": n,
        "total_distance_px": total_dist,
        "net_displacement_px": net_disp,
        "avg_speed_px_per_frame": avg_speed,
        "max_step_px": max_step,
    }
