import os
import numpy as np
import matplotlib.pyplot as plt

from backend.tracking.frame_provider import FrameProvider

from .io_index import load_analysis_index, group_by_track
from .roi import extract_complex_patch, build_complex_stack_no_recenter
from .metrics import compute_patch_metrics, gaussian_mass_abs2, track_motion_metrics
from .plots import save_population_plots
from .trajectory_gif import make_trajectory_gif


def _save_tracks_csv(path: str, rows: list[dict]):
    if not rows:
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
            f.write(",".join(str(r.get(k, np.nan)) for k in keys) + "\n")


def run_trackwise_analysis(
    analysis_index_path: str,
    output_folder: str,
    fft_settings: dict,
    roi_radius: int,
    export_timeseries_top_n: int = 0,
    log_fn=None,
    status_fn=None,
):
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

    export_set = set()
    if export_timeseries_top_n and export_timeseries_top_n > 0:
        lengths = [(tid, len(ix)) for tid, ix in groups]
        lengths.sort(key=lambda z: z[1], reverse=True)
        export_set = {tid for tid, _ in lengths[: int(export_timeseries_top_n)]}
        log(f"Will export timeseries for top {len(export_set)} longest tracks.")

    rows = []

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

    for k, (tid, ix) in enumerate(groups):
        if k % 10 == 0:
            status(f"Analyzing track {k+1}/{n_tracks}...")

        tt = t_all[ix].astype(np.int32)
        xx = x_all[ix].astype(np.float32)
        yy = y_all[ix].astype(np.float32)
        sc = det_score_all[ix].astype(np.float32) if det_score_all is not None else None

        order = np.argsort(tt, kind="stable")
        tt, xx, yy = tt[order], xx[order], yy[order]
        sc = sc[order] if sc is not None else None

        n = int(len(tt))
        if n == 0:
            continue

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
            mass_t[i] = float(gaussian_mass_abs2(patch))
            amp_mean_t[i] = float(m["amp_mean"])
            amp_max_t[i] = float(m["amp_max"])
            ph_std_t[i] = float(m["ph_std"])
            valid += 1

        valid_frac = float(valid / n) if n > 0 else 0.0

        mass_valid = mass_t[np.isfinite(mass_t)]
        if mass_valid.size == 0:
            mass_med = np.nan
            mass_mean = np.nan
            mass_std = np.nan
        else:
            mass_med = float(np.median(mass_valid))
            mass_mean = float(np.mean(mass_valid))
            mass_std = float(np.std(mass_valid))

        amp_mean_valid = amp_mean_t[np.isfinite(amp_mean_t)]
        amp_mean_med = float(np.median(amp_mean_valid)) if amp_mean_valid.size else np.nan
        amp_mean_mean = float(np.mean(amp_mean_valid)) if amp_mean_valid.size else np.nan

        motion = track_motion_metrics(tt, xx, yy)

        if tid in export_set:
            track_dir = os.path.join(timeseries_dir, "analysis_particles", f"track_{tid:05d}")
            os.makedirs(track_dir, exist_ok=True)

            cstack = build_complex_stack_no_recenter(provider, tt, xx, yy, roi_radius)

            if cstack is not None:
                mean_c = np.mean(cstack, axis=0)
                re = np.real(mean_c)
                im = np.imag(mean_c)
                amp = np.abs(mean_c)
                phase = np.angle(mean_c)

                np.savez(
                    os.path.join(track_dir, "mean_complex_particle.npz"),
                    mean_complex=mean_c.astype(np.complex64),
                    roi_radius=np.int32(roi_radius),
                    n_frames=np.int32(cstack.shape[0]),
                )

                fig, axs = plt.subplots(2, 2, figsize=(6, 6))
                axs = axs.ravel()
                ims = [
                    axs[0].imshow(re, cmap="viridis"),
                    axs[1].imshow(im, cmap="viridis"),
                    axs[2].imshow(amp, cmap="viridis"),
                    axs[3].imshow(phase, cmap="viridis"),
                ]
                titles = ["Re(E)", "Im(E)", "|E|", "Phase"]
                for ax, im_, title in zip(axs, ims, titles):
                    ax.set_title(title)
                    ax.axis("off")
                    fig.colorbar(im_, ax=ax, fraction=0.046)
                fig.suptitle(f"Track {tid} – mean complex particle (no recenter)")
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig.savefig(os.path.join(track_dir, "mean_particle_complex_2x2.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)

            np.savez(
                os.path.join(track_dir, "timeseries.npz"),
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

    tracks_csv = os.path.join(output_folder, "analysis_tracks.csv")
    _save_tracks_csv(tracks_csv, rows)
    log(f"Saved: {tracks_csv}")

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

    plots_dir = os.path.join(output_folder, "analysis_plots")
    save_population_plots(plots_dir, rows, log)


    # Create trajectory GIF
    gif_path = make_trajectory_gif(
        analysis_index=idx,
        output_folder=output_folder,
        frame_provider=provider,
        fps=10,
        trail=30,
    )

    log(f"Saved trajectory GIF: {gif_path}")


    status("Analysis done.")
    return {
        "tracks_csv": tracks_csv,
        "tracks_npz": tracks_npz,
        "plots_dir": plots_dir,
    }
