import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def scatter_xy(xs, ys, out_png, title, xlabel, ylabel, logx=False, logy=False):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if xs.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, s=10, alpha=0.35, linewidths=0)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def hexbin_xy(xs, ys, out_png, title, xlabel, ylabel, gridsize=40, logx=False, logy=False):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if xs.size == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    hb = ax.hexbin(xs, ys, gridsize=gridsize, mincnt=1)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    fig.colorbar(hb, ax=ax, label="count")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def binned_trend(xs, ys, out_png, title, xlabel, ylabel, nbins=20, logx=False, logy=False):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[m], ys[m]
    if xs.size == 0:
        return

    bx = np.log10(xs[xs > 0]) if logx else xs
    if logx:
        xs = xs[xs > 0]
        ys = ys[: xs.size]  # safe enough given your usage; can tighten later

    edges = np.linspace(bx.min(), bx.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    med = np.full(nbins, np.nan)
    q25 = np.full(nbins, np.nan)
    q75 = np.full(nbins, np.nan)

    for i in range(nbins):
        sel = (bx >= edges[i]) & (bx < edges[i + 1])
        if np.count_nonzero(sel) < 5:
            continue
        yy = ys[sel]
        med[i] = np.median(yy)
        q25[i] = np.percentile(yy, 25)
        q75[i] = np.percentile(yy, 75)

    good = np.isfinite(med)
    if not np.any(good):
        return

    xplot = 10**centers[good] if logx else centers[good]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, s=8, alpha=0.2, linewidths=0)
    ax.plot(xplot, med[good], lw=2.0, label="binned median")
    ax.fill_between(xplot, q25[good], q75[good], alpha=0.25, label="IQR")
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_population_plots(plot_dir: str, rows: list[dict], log):
    if not rows:
        return

    os.makedirs(plot_dir, exist_ok=True)

    def arr(key):
        a = np.asarray([r.get(key, np.nan) for r in rows], dtype=np.float32)
        return a[np.isfinite(a)]

    def hist_with_density(x, fname, title, xlabel, bins=40, kde=True):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size < 5:
            return

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(x, bins=bins, density=True, alpha=0.6, edgecolor="white", linewidth=0.5, label="hist")

        xs = np.linspace(x.min(), x.max(), 500)
        if kde:
            kde_est = stats.gaussian_kde(x)
            ax.plot(xs, kde_est(xs), lw=2.0, label="KDE")

        mean = np.mean(x)
        median = np.median(x)
        ax.axvline(mean, ls="--", lw=1.2, label="mean")
        ax.axvline(median, ls=":", lw=1.2, label="median")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("probability density")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        p = os.path.join(plot_dir, fname)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"Saved plot: {p}")

    hist_with_density(arr("mass_median"), "hist_mass_median.png", "Particle mass proxy (median per track)", "mass proxy")
    hist_with_density(arr("mass_mean"), "hist_mass_mean.png", "Particle mass proxy (mean per track)", "mass proxy")
    hist_with_density(arr("n_frames"), "hist_n_frames.png", "Track length distribution", "n_frames")
    hist_with_density(arr("avg_speed_px_per_frame"), "hist_avg_speed.png", "Average speed distribution", "avg_speed (px/frame)")
    hist_with_density(arr("total_distance_px"), "hist_total_distance.png", "Total distance distribution", "total_distance (px)")

    xs = []
    ys = []
    for r in rows:
        nf = r.get("n_frames", np.nan)
        mm = r.get("mass_median", np.nan)
        if np.isfinite(nf) and np.isfinite(mm):
            xs.append(nf)
            ys.append(mm)

    if xs:
        scatter_xy(xs, ys, os.path.join(plot_dir, "scatter_mass_median_vs_length.png"),
                   "Mass median vs track length", "n_frames", "mass_median", logx=False, logy=True)

        hexbin_xy(xs, ys, os.path.join(plot_dir, "hexbin_mass_median_vs_length.png"),
                  "Mass median vs track length (density)", "n_frames", "mass_median",
                  gridsize=45, logx=False, logy=True)

        binned_trend(xs, ys, os.path.join(plot_dir, "trend_mass_median_vs_length.png"),
                    "Mass median vs track length (trend)", "n_frames", "mass_median",
                    nbins=20, logx=False, logy=True)
