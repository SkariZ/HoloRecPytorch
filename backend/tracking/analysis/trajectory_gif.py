import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from backend.image_utils import save_gif, cleanup_frames


def make_trajectory_gif(
    analysis_index: dict,
    output_folder: str,
    frame_provider,
    fps: int = 10,
    trail: int = 20,
    max_tracks: int | None = None,
):
    """
    Create a GIF showing all active tracks over time.

    Args:
        analysis_index (dict):
            Output of load_analysis_index()
            Must contain: t, x, y, track_id
        output_folder (str):
            Analysis output folder (GIF saved into analysis_plots/)
        frame_provider:
            FrameProvider instance (for background image)
        fps (int):
            GIF frame rate
        trail (int):
            How many past frames to show as fading trail
        max_tracks (int | None):
            Optional limit for number of tracks (for very dense scenes)
    """

    plots_dir = os.path.join(output_folder, "analysis_plots")
    frames_dir = os.path.join(plots_dir, "_tmp_traj_frames")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    t = analysis_index["t"]
    x = analysis_index["x"]
    y = analysis_index["y"]
    tid = analysis_index["track_id"]

    # Sort by time
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    y = y[order]
    tid = tid[order]

    unique_t = np.unique(t)

    # Optional track limiting
    if max_tracks is not None:
        keep_ids = np.unique(tid)[:max_tracks]
        mask = np.isin(tid, keep_ids)
        t, x, y, tid = t[mask], x[mask], y[mask], tid[mask]

    # Color per track (stable)
    track_ids = np.unique(tid)
    colors = {
        k: plt.cm.tab20(i % 20)
        for i, k in enumerate(track_ids)
    }

    for i, ti in enumerate(unique_t):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.invert_yaxis()

        # Background image (optional but very nice)
        try:
            bg = frame_provider.get_frame_np(int(ti))
            ax.imshow(bg, cmap="gray")
        except Exception:
            pass

        tmin = ti - trail
        mask = (t >= tmin) & (t <= ti)

        for k in np.unique(tid[mask]):
            m = mask & (tid == k)
            if not np.any(m):
                continue

            age = ti - t[m]
            alpha = np.clip(1.0 - age / max(1, trail), 0.1, 1.0)

            ax.plot(
                x[m],
                y[m],
                "-",
                color=colors[k],
                alpha=alpha[-1] if len(alpha) else 1.0,
                linewidth=1.2,
            )

            ax.scatter(
                x[m][-1],
                y[m][-1],
                s=12,
                color=colors[k],
                alpha=1.0,
            )

        ax.set_title(f"Frame {int(ti)}")
        ax.axis("off")

        frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
        fig.savefig(frame_path, dpi=120, bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig)

    # ---- Build GIF ----
    gif_path = os.path.join(plots_dir, "trajectories_active.gif")
    save_gif(
        folder=frames_dir,
        savefile=gif_path,
        duration=int(1000 / fps),
        loop=0,
    )

    # ---- Cleanup ----
    shutil.rmtree(frames_dir, ignore_errors=True)

    return gif_path
