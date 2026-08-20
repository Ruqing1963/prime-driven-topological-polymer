# -*- coding: utf-8 -*-
"""
Regenerate all paper figures (vector PDF) from the data files in ../data/.

  Fig. 1  conformation      <- conformation_coords.npy, conformation_kbend.npy
  Fig. 2  decisive 40k      <- decisive_40k_results.csv
  Fig. 3  decomposition     <- decomposition_results.csv
  Fig. 4  finite-size       <- finite_size_results.csv

Usage:  python3 make_figures.py
Dependencies: numpy, scipy, matplotlib
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIGS = os.path.join(HERE, "..", "figures")
os.makedirs(FIGS, exist_ok=True)

C_TRUE, C_S1, C_S3 = "#c1272d", "#0868ac", "#238b45"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "pdf.fonttype": 42})


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2) + 1e-12
    d = (a.mean() - b.mean()) / pooled
    return p, d


# ---------------------------------------------------------------- Fig. 1
def fig1_conformation():
    r = np.load(os.path.join(DATA, "conformation_coords.npy"))
    k = np.load(os.path.join(DATA, "conformation_kbend.npy"))
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    c = np.log10(np.clip(k, 1e-3, None))
    ax.plot(r[:, 0], r[:, 1], r[:, 2], color="0.8", lw=0.3, alpha=0.5, zorder=1)
    sc = ax.scatter(r[:, 0], r[:, 1], r[:, 2], c=c, cmap="coolwarm", s=4, alpha=0.75, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(r"$\log_{10}\,k_i$   (low = flexible core,  high = rigid rod)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ext = np.array([r[:, j].max() - r[:, j].min() for j in range(3)])
    try: ax.set_box_aspect(ext)
    except Exception: pass
    ax.view_init(elev=18, azim=-60)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_conformation.pdf"))
    plt.close(fig)
    print("Fig. 1 written.")


# ---------------------------------------------------------------- helpers for CSV
def load_rho(path):
    """model -> list of spearman_rho  (schema: model,run,spearman_rho)."""
    d = {}
    for line in open(path):
        p = line.strip().split(",")
        if len(p) == 3 and p[0] != "model":
            d.setdefault(p[0], []).append(float(p[2]))
    return {m: np.array(v) for m, v in d.items()}


def load_rho_dr(path):
    """model -> rho, dR arrays (schema: model,run,spearman_rho,dR)."""
    d = {}
    for line in open(path):
        p = line.strip().split(",")
        if len(p) == 4 and p[0] != "model":
            d.setdefault(p[0], {"rho": [], "dR": []})
            d[p[0]]["rho"].append(float(p[2])); d[p[0]]["dR"].append(float(p[3]))
    return {m: {k: np.array(v) for k, v in dd.items()} for m, dd in d.items()}


def load_nscale(path):
    """model -> N -> rho,dR (schema: model,N,run,spearman_rho,dR)."""
    d = {}
    for line in open(path):
        p = line.strip().split(",")
        if len(p) == 5 and p[0] != "model":
            m, N = p[0], int(p[1])
            d.setdefault(m, {}).setdefault(N, {"rho": [], "dR": []})
            d[m][N]["rho"].append(float(p[3])); d[m][N]["dR"].append(float(p[4]))
    return d


def _box(ax, groups, colors, labels, ylabel):
    bp = ax.boxplot(groups, positions=range(1, len(groups) + 1), widths=0.55,
                    patch_artist=True, showfliers=False, medianprops=dict(color="k", lw=1.6))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for pos, (g, col) in enumerate(zip(groups, colors), start=1):
        x = pos + (rng.random(len(g)) - 0.5) * 0.20
        ax.scatter(x, g, s=28, color=col, edgecolor="k", lw=0.5, alpha=0.85, zorder=3)
    ax.set_xticks(range(1, len(groups) + 1)); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------- Fig. 2
def fig2_decisive():
    d = load_rho_dr(os.path.join(DATA, "decisive_40k_results.csv"))
    T, S1 = d["True"], d["S1"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    _box(ax[0], [T["rho"], S1["rho"]], [C_TRUE, C_S1], ["True primes", "S1 shuffle"],
         r"Spearman $\rho$ (stiffness vs radius)")
    p, dd = welch(T["rho"], S1["rho"]); ax[0].axhline(0, color="gray", ls="--", lw=1)
    ax[0].set_title(f"(a)  $p={p:.1e}$,  $d={dd:+.2f}$")
    _box(ax[1], [T["dR"], S1["dR"]], [C_TRUE, C_S1], ["True primes", "S1 shuffle"],
         r"$\Delta R$ (core--shell)")
    p2, dd2 = welch(T["dR"], S1["dR"]); ax[1].axhline(0, color="gray", ls="--", lw=1)
    ax[1].set_title(f"(b)  $p={p2:.2f}$,  $d={dd2:+.2f}$")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_decisive.pdf")); plt.close(fig)
    print(f"Fig. 2 written. rho: p={p:.2e} d={dd:+.2f} | dR: p={p2:.2f}")


# ---------------------------------------------------------------- Fig. 3
def fig3_decomposition():
    d = load_rho(os.path.join(DATA, "decomposition_results.csv"))
    T, S1, S3 = d["True"], d["S1"], d["S3"]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    _box(ax, [T, S1, S3], [C_TRUE, C_S1, C_S3],
         ["True primes\n(ordered)", "S1 shuffle\n(real marginal)", "S3 Cramer\n(random marginal)"],
         r"Spearman $\rho$ (stiffness vs radius)")
    ax.axhline(0, color="gray", ls="--", lw=1)
    p_os, d_os = welch(T, S1); p_cb, d_cb = welch(T, S3)
    ax.set_title("Ordering vs marginal decomposition")
    ax.text(0.03, 0.97, f"True vs S1 (ordering): $p={p_os:.1e}$, $d={d_os:+.2f}$\n"
                        f"True vs S3 (combined): $p={p_cb:.1e}$, $d={d_cb:+.2f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_decomposition.pdf")); plt.close(fig)
    print(f"Fig. 3 written. TruevsS1 p={p_os:.2e} | S1vsS3 p={welch(S1,S3)[0]:.2f}")


# ---------------------------------------------------------------- Fig. 4
def fig4_finite_size():
    d = load_nscale(os.path.join(DATA, "finite_size_results.csv"))
    Ns = sorted(d["True"].keys())

    def ms(m, key):
        xs, ys, es = [], [], []
        for N in Ns:
            v = np.array(d[m][N][key])
            if len(v) >= 2:
                xs.append(N); ys.append(v.mean()); es.append(v.std(ddof=1) / np.sqrt(len(v)))
        return xs, ys, es

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, key, ylab, tag in [(ax[0], "rho", r"Spearman $\rho$ (stiffness vs radius)", "(a)"),
                              (ax[1], "dR", r"$\Delta R$ (core--shell)", "(b)")]:
        for m, col in [("True", C_TRUE), ("S1", C_S1)]:
            xs, ys, es = ms(m, key)
            a.errorbar(xs, ys, yerr=es, marker="o", ms=7, lw=2, capsize=4, color=col, label=m)
        a.axhline(0, color="gray", ls="--", lw=1); a.set_xscale("log")
        a.set_xticks(Ns); a.set_xticklabels([str(N) for N in Ns])
        a.set_xlabel("N"); a.set_ylabel(ylab); a.grid(alpha=0.3); a.legend(); a.set_title(tag)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig4_finite_size.pdf")); plt.close(fig)
    print("Fig. 4 written.")


if __name__ == "__main__":
    fig1_conformation()
    fig2_decisive()
    fig3_decomposition()
    fig4_finite_size()
    print("All figures ->", os.path.abspath(FIGS))
