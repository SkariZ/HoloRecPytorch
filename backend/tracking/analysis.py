# backend/tracking/analysis.py
import os
import numpy as np
import matplotlib.pyplot as plt

from backend.tracking.frame_provider import FrameProvider


# -----------------------------
# Loading / grouping utilities
# -----------------------------
def load_analysis_index(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {
        "track_id": data["track_id"].astype(np.int32),
        "t": data["t"].astype(np.int32),
        "x": data["x"].astype(np.float32),
        "y": data["y"].astype(np.float32),
        "score": data["score"].astype(np.float32) if "score" in data.files else None,
        "input_path": str(data["input_path"]) if "input_path" in data.files else None,
        "roi_radius": int(data["roi_radius"]) if "roi_radius" in data.files else None,
        "channel": str(data["channel"]) if "channel" in data.files else None,
    }
    return out


def group_by_track(track_id: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """
    Returns a list of (tid, indices) for each unique track_id.
    """
    track_id = track_id.astype(np.int32, copy=False)
    order = np.argsort(track_id, kind="stable")
    tid_sorted = track_id[order]

    if tid_sorted.size == 0:
        return []

    cuts = np.flatnonzero(np.diff(tid_sorted)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, tid_sorted.size]

    groups = []
    for s, e in zip(starts, ends):
        tid = int(tid_sorted[s])
        idx = order[s:e]
        groups.append((tid, idx))
    return groups


# -----------------------------
# Complex ROI extraction
# -----------------------------
def extract_complex_patch(field: np.ndarray, x: float, y: float, r: int):
    """
    Extract (2r+1)x(2r+1) complex patch centered at (x,y).
    Returns None if out of bounds.
    """
    x0 = int(round(float(x)))
    y0 = int(round(float(y)))
    x1, x2 = x0 - r, x0 + r + 1
    y1, y2 = y0 - r, y0 + r + 1
    H, W = field.shape
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        return None
    return field[y1:y2, x1:x2]


# -----------------------------
# Metrics
# -----------------------------
def compute_patch_metrics(patch: np.ndarray) -> dict:
    """
    Basic per-frame metrics from complex patch.

    NOTE: phase stats here are naive (wrap at +/-pi). Fine for start.
    If phase matters, we can swap to circular stats later.
    """
    amp = np.abs(patch).astype(np.float32)
    ph = np.angle(patch).astype(np.float32)

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


def compute_mass_from_patch(patch: np.ndarray) -> float:
    """
    Placeholder for your 'mass' definition.

    Replace with your physical model when ready.

    A reasonable first proxy (dimensionless):
      mass ~ sum(|E|) over ROI
    or mass ~ sum(|E|^2) over ROI

    Start with sum(|E|) to keep it simple.
    """
    amp = np.abs(patch).astype(np.float32)
    return float(amp.sum())


def track_motion_metrics(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """
    Motion metrics in pixels and frames.
    Assumes arrays are time-ordered.
    """
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


# -----------------------------
# Main entry point
# -----------------------------
def run_trackwise_analysis(
    analysis_index_path: str,
    output_folder: str,
    fft_settings: dict,
    roi_radius: int,
    export_timeseries_top_n: int = 0,
    log_fn=None,
    status_fn=None,
):
    """
    Track-by-track analysis:

    - Loads analysis_index.npz (flat samples)
    - Groups by track_id
    - For each track:
        - Extract complex ROI per timestep
        - Compute mass_t + other patch metrics per frame (in memory)
        - Aggregate to per-track summary
    - Saves:
        - analysis_tracks.csv (one row per track)
        - analysis_tracks.npz (arrays)
        - analysis_plots/*.png
        - optionally timeseries for top N longest tracks
    """

    def log(msg: str):
        if log_fn:
            log_fn(msg)

    def status(msg: str):
        if status_fn:
            status_fn(msg)

    idx = load_analysis_index(analysis_index_path)

    input_path = idx.get("input_path")
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError(f"Input .npy not found: {input_path}")

    os.makedirs(output_folder, exist_ok=True)
    
    timeseries_dir = os.path.join(output_folder, "analysis_timeseries_top")
    os.makedirs(timeseries_dir, exist_ok=True)

    # Build provider
    full_data = np.load(input_path, mmap_mode="r")
    provider = FrameProvider(full_data, **fft_settings)

    track_id = idx["track_id"]
    t_all = idx["t"]
    x_all = idx["x"]
    y_all = idx["y"]
    det_score_all = idx["score"]

    groups = group_by_track(track_id)
    n_tracks = len(groups)
    log(f"Trackwise analysis: tracks={n_tracks}, roi_radius={roi_radius}")

    # Decide which tracks get timeseries exported (longest N)
    export_set = set()
    if export_timeseries_top_n and export_timeseries_top_n > 0:
        lengths = [(tid, len(ix)) for tid, ix in groups]
        lengths.sort(key=lambda z: z[1], reverse=True)
        export_set = {tid for tid, _ in lengths[: int(export_timeseries_top_n)]}
        log(f"Will export timeseries for top {len(export_set)} longest tracks.")

    # Per-track summary rows
    rows = []

    # Also keep arrays for NPZ export
    out_tid = []
    out_n = []
    out_mass_median = []
    out_mass_mean = []
    out_mass_std = []
    out_valid_frac = []
    out_total_dist = []
    out_net_disp = []
    out_avg_speed = []
    out_max_step = []

    out_amp_mean_med = []
    out_amp_mean_mean = []

    # Loop tracks
    for k, (tid, ix) in enumerate(groups):
        if k % 10 == 0:
            status(f"Analyzing track {k+1}/{n_tracks}...")

        # Extract track samples
        tt = t_all[ix].astype(np.int32)
        xx = x_all[ix].astype(np.float32)
        yy = y_all[ix].astype(np.float32)
        sc = det_score_all[ix].astype(np.float32) if det_score_all is not None else None

        # Sort within track by time (important!)
        order = np.argsort(tt, kind="stable")
        tt = tt[order]
        xx = xx[order]
        yy = yy[order]
        sc = sc[order] if sc is not None else None

        n = int(len(tt))
        if n == 0:
            continue

        # Per-frame arrays (kept in memory for this track only)
        mass_t = np.full(n, np.nan, dtype=np.float32)
        amp_mean_t = np.full(n, np.nan, dtype=np.float32)
        amp_max_t = np.full(n, np.nan, dtype=np.float32)
        ph_std_t = np.full(n, np.nan, dtype=np.float32)

        valid = 0

        for i in range(n):
            try:
                field = provider.get_complex_frame_np(int(tt[i]))
            except Exception:
                continue

            patch = extract_complex_patch(field, float(xx[i]), float(yy[i]), int(roi_radius))
            if patch is None:
                continue

            m = compute_patch_metrics(patch)
            mass_t[i] = float(compute_mass_from_patch(patch))
            amp_mean_t[i] = float(m["amp_mean"])
            amp_max_t[i] = float(m["amp_max"])
            ph_std_t[i] = float(m["ph_std"])
            valid += 1

        valid_frac = float(valid / n) if n > 0 else 0.0

        # Aggregate mass metrics
        mass_valid = mass_t[np.isfinite(mass_t)]
        if mass_valid.size == 0:
            mass_med = np.nan
            mass_mean = np.nan
            mass_std = np.nan
        else:
            mass_med = float(np.median(mass_valid))
            mass_mean = float(np.mean(mass_valid))
            mass_std = float(np.std(mass_valid))

        # Some amplitude aggregates (example)
        amp_mean_valid = amp_mean_t[np.isfinite(amp_mean_t)]
        amp_mean_med = float(np.median(amp_mean_valid)) if amp_mean_valid.size else np.nan
        amp_mean_mean = float(np.mean(amp_mean_valid)) if amp_mean_valid.size else np.nan

        # Motion metrics from coordinates (even if ROI invalid sometimes)
        motion = track_motion_metrics(tt, xx, yy)

        # Save optional timeseries for this track (only for selected tracks)
        if tid in export_set:
            track_dir = os.path.join(timeseries_dir, "analysis_particles", f"track_{tid:05d}")
            os.makedirs(track_dir, exist_ok=True)

            # ---- complex mean particle (no recenter) ----
            cstack = build_complex_stack_no_recenter(
                provider, tt, xx, yy, roi_radius
            )

            if cstack is not None:
                mean_c = np.mean(cstack, axis=0)  # complex mean

                re = np.real(mean_c)
                im = np.imag(mean_c)
                amp = np.abs(mean_c)
                phase = np.angle(mean_c)

                # save complex mean for later (very useful!)
                np.savez(
                    os.path.join(track_dir, "mean_complex_particle.npz"),
                    mean_complex=mean_c.astype(np.complex64),
                    roi_radius=np.int32(roi_radius),
                    n_frames=np.int32(cstack.shape[0]),
                )

                # ---- 2x2 plot ----
                fig, axs = plt.subplots(2, 2, figsize=(6, 6))
                axs = axs.ravel()

                ims = [
                    axs[0].imshow(re, cmap='viridis'),
                    axs[1].imshow(im, cmap='viridis'),
                    axs[2].imshow(amp, cmap='viridis'),
                    axs[3].imshow(phase, cmap='viridis'),
                ]

                titles = ["Re(E)", "Im(E)", "|E|", "Phase"]
                for ax, im_, title in zip(axs, ims, titles):
                    ax.set_title(title)
                    ax.axis("off")
                    fig.colorbar(im_, ax=ax, fraction=0.046)

                fig.suptitle(f"Track {tid} – mean complex particle (no recenter)")
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                fig.savefig(
                    os.path.join(track_dir, "mean_particle_complex_2x2.png"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

            # ---- save timeseries (optional) ----
            ts_path = os.path.join(track_dir, "timeseries.npz")
            np.savez(
                ts_path,
                track_id=np.int32(tid),
                t=tt, x=xx, y=yy,
                det_score=sc if sc is not None else np.zeros_like(xx),
                mass=mass_t,
                amp_mean=amp_mean_t,
                amp_max=amp_max_t,
                ph_std=ph_std_t,
                roi_radius=np.int32(roi_radius),
            )

            log(f"Saved particle folder: {track_dir}")

        # Build track summary row
        row = {
            "track_id": int(tid),
            "n_frames": int(motion["n_frames"]),
            "valid_fraction": float(valid_frac),

            "mass_median": mass_med,
            "mass_mean": mass_mean,
            "mass_std": mass_std,

            "amp_mean_median": amp_mean_med,
            "amp_mean_mean": amp_mean_mean,

            "total_distance_px": float(motion["total_distance_px"]),
            "net_displacement_px": float(motion["net_displacement_px"]),
            "avg_speed_px_per_frame": float(motion["avg_speed_px_per_frame"]),
            "max_step_px": float(motion["max_step_px"]),
        }
        rows.append(row)

        # arrays for NPZ
        out_tid.append(int(tid))
        out_n.append(int(motion["n_frames"]))
        out_valid_frac.append(float(valid_frac))
        out_mass_median.append(mass_med)
        out_mass_mean.append(mass_mean)
        out_mass_std.append(mass_std)
        out_amp_mean_med.append(amp_mean_med)
        out_amp_mean_mean.append(amp_mean_mean)
        out_total_dist.append(float(motion["total_distance_px"]))
        out_net_disp.append(float(motion["net_displacement_px"]))
        out_avg_speed.append(float(motion["avg_speed_px_per_frame"]))
        out_max_step.append(float(motion["max_step_px"]))

    # Save per-track table
    tracks_csv = os.path.join(output_folder, "analysis_tracks.csv")
    _save_tracks_csv(tracks_csv, rows)
    log(f"Saved: {tracks_csv}")

    # Save NPZ version
    tracks_npz = os.path.join(output_folder, "analysis_tracks.npz")
    np.savez(
        tracks_npz,
        track_id=np.asarray(out_tid, dtype=np.int32),
        n_frames=np.asarray(out_n, dtype=np.int32),
        valid_fraction=np.asarray(out_valid_frac, dtype=np.float32),

        mass_median=np.asarray(out_mass_median, dtype=np.float32),
        mass_mean=np.asarray(out_mass_mean, dtype=np.float32),
        mass_std=np.asarray(out_mass_std, dtype=np.float32),

        amp_mean_median=np.asarray(out_amp_mean_med, dtype=np.float32),
        amp_mean_mean=np.asarray(out_amp_mean_mean, dtype=np.float32),

        total_distance_px=np.asarray(out_total_dist, dtype=np.float32),
        net_displacement_px=np.asarray(out_net_disp, dtype=np.float32),
        avg_speed_px_per_frame=np.asarray(out_avg_speed, dtype=np.float32),
        max_step_px=np.asarray(out_max_step, dtype=np.float32),

        roi_radius=np.int32(roi_radius),
        input_path=input_path,
    )
    log(f"Saved: {tracks_npz}")

    # Population plots
    plots_dir = os.path.join(output_folder, "analysis_plots")
    os.makedirs(plots_dir, exist_ok=True)
    _save_population_plots(plots_dir, rows, log)

    status("Analysis done.")
    return {
        "tracks_csv": tracks_csv,
        "tracks_npz": tracks_npz,
        "plots_dir": plots_dir,
    }


# -----------------------------
# Saving helpers
# -----------------------------
def _save_tracks_csv(path: str, rows: list[dict]):
    if not rows:
        # still write header for consistency
        header = (
            "track_id,n_frames,valid_fraction,"
            "mass_median,mass_mean,mass_std,"
            "amp_mean_median,amp_mean_mean,"
            "total_distance_px,net_displacement_px,avg_speed_px_per_frame,max_step_px"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n")
        return

    keys = [
        "track_id", "n_frames", "valid_fraction",
        "mass_median", "mass_mean", "mass_std",
        "amp_mean_median", "amp_mean_mean",
        "total_distance_px", "net_displacement_px", "avg_speed_px_per_frame", "max_step_px",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            vals = []
            for k in keys:
                v = r.get(k, np.nan)
                if v is None:
                    v = np.nan
                vals.append(str(v))
            f.write(",".join(vals) + "\n")

def build_stack_no_recenter(provider, tt, xx, yy, roi_radius, intensity_mode="abs2"):
    """
    Returns stack (n_valid, H, W) of per-frame particle ROIs, without recentering.
    """
    stack = []
    for t, x, y in zip(tt, xx, yy):
        field = provider.get_complex_frame_np(int(t))
        patch = extract_complex_patch(field, float(x), float(y), int(roi_radius))
        if patch is None:
            continue

        if intensity_mode == "abs2":
            img = (np.abs(patch) ** 2).astype(np.float32)
        else:
            img = np.abs(patch).astype(np.float32)

        stack.append(img)

    if not stack:
        return None
    return np.stack(stack, axis=0)

def build_complex_stack_no_recenter(provider, tt, xx, yy, roi_radius):
    """
    Returns stack (n_valid, H, W) of complex ROIs, without recentering.
    """
    stack = []
    for t, x, y in zip(tt, xx, yy):
        field = provider.get_complex_frame_np(int(t))
        patch = extract_complex_patch(field, float(x), float(y), int(roi_radius))
        if patch is None:
            continue
        stack.append(patch.astype(np.complex64))

    if not stack:
        return None
    return np.stack(stack, axis=0)

def _save_population_plots(plot_dir: str, rows: list[dict], log):
    if not rows:
        return

    def arr(key):
        a = np.asarray([r.get(key, np.nan) for r in rows], dtype=np.float32)
        return a[np.isfinite(a)]

    # basic histograms
    def hist(x, fname, title, xlabel):
        x = np.asarray(x)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        plt.figure()
        plt.hist(x, bins=40, color="#007acc", alpha=0.8, density=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        p = os.path.join(plot_dir, fname)
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"Saved plot: {p}")

    hist(arr("mass_median"), "hist_mass_median.png", "Track mass median distribution", "mass_median")
    hist(arr("mass_mean"), "hist_mass_mean.png", "Track mass mean distribution", "mass_mean")
    hist(arr("n_frames"), "hist_track_length.png", "Track length distribution", "n_frames")
    hist(arr("avg_speed_px_per_frame"), "hist_avg_speed.png", "Average speed distribution", "avg_speed (px/frame)")
    hist(arr("total_distance_px"), "hist_total_distance.png", "Total distance distribution", "total_distance (px)")

    # simple scatter: mass vs length (optional)
    x = arr("n_frames")
    y = arr("mass_median")
    if x.size and y.size:
        # align by filtering both from rows (simple approach)
        xs = []
        ys = []
        for r in rows:
            nf = r.get("n_frames", np.nan)
            mm = r.get("mass_median", np.nan)
            if np.isfinite(nf) and np.isfinite(mm):
                xs.append(nf)
                ys.append(mm)
        if xs:
            plt.figure()
            plt.scatter(xs, ys, s=10)
            plt.title("Mass median vs track length")
            plt.xlabel("n_frames")
            plt.ylabel("mass_median")
            p = os.path.join(plot_dir, "scatter_mass_median_vs_length.png")
            plt.savefig(p, dpi=150, bbox_inches="tight")
            plt.close()
            log(f"Saved plot: {p}")