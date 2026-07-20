# -*- coding: utf-8 -*-
"""Regenerate the S4/S5 scissors figure (Fig. 5) from the ensemble CSVs.

Reads (from ../data/ or the working dir):
  true_vs_s1_results.csv          -> True and S1 reference bands
  block_shuffle_results.csv       -> S4 arms (S4_L10/50/100/500)
  block_local_scramble_results.csv-> S5 arms (S5_L10/50/100/500)
Writes ../figures/fig5_scissors.pdf.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8})
LS = np.array([10, 50, 100, 500])
XI, XI_LO, XI_HI = 355.0, 180.0, 490.0     # half-retention crossing + 68% bootstrap CI


def _find(name):
    for c in (name, os.path.join("..", "data", name), os.path.join("data", name)):
        if os.path.exists(c):
            return c
    raise FileNotFoundError(name)

def load(name):
    d = {}
    for line in open(_find(name)).read().splitlines()[1:]:
        p = line.strip().split(",")
        if len(p) == 3:
            d.setdefault(p[0], []).append(float(p[2]))
    return {k: np.array(v) for k, v in d.items()}

def ms(d, pre):
    m = np.array([d[f"{pre}{L}"].mean() for L in LS])
    e = np.array([d[f"{pre}{L}"].std(ddof=1) / np.sqrt(len(d[f"{pre}{L}"])) for L in LS])
    return m, e


def main():
    base = load("true_vs_s1_results.csv")
    s4 = load("block_shuffle_results.csv")
    s5 = load("block_local_scramble_results.csv")
    T, S1 = base["True"], base["S1"]
    m4, e4 = ms(s4, "S4_L"); m5, e5 = ms(s5, "S5_L")

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    xlo, xhi = 6, 900
    ax.axvspan(XI_LO, XI_HI, color="0.5", alpha=0.10, zorder=0)
    for name, a, c in [("True", T, "#c1272d"), ("S1", S1, "#0868ac")]:
        m = a.mean(); s = a.std(ddof=1) / np.sqrt(len(a))
        ax.axhline(m, color=c, ls="--", lw=1.4, zorder=1)
        ax.fill_between([xlo, xhi], m - s, m + s, color=c, alpha=0.13, zorder=0)
    ax.errorbar(LS, m4, yerr=e4, marker="s", ms=7, lw=1.8, capsize=3, color="#6a3d9a",
                zorder=5, label="S4  block shuffle (destroy global)")
    ax.errorbar(LS, m5, yerr=e5, marker="o", ms=7, lw=1.8, capsize=3, color="#2ca25f",
                zorder=5, label="S5  local scramble (destroy local)")
    ax.text(700, T.mean() - 0.004, "True", color="#c1272d", fontsize=9, ha="center", va="top")
    ax.text(700, S1.mean() + 0.003, "S1", color="#0868ac", fontsize=9, ha="center", va="bottom")
    ax.text(np.sqrt(XI_LO * XI_HI), -0.052, r"$\xi\approx3.5\times10^{2}$",
            fontsize=9, ha="center", color="0.3")
    ax.set_xscale("log"); ax.set_xlim(xlo, xhi); ax.set_ylim(-0.058, 0.05)
    ax.set_xlabel(r"block length $L$ (gaps)")
    ax.set_ylabel(r"Spearman $\rho$ (stiffness vs.\ radius)")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    out = os.path.join("..", "figures", "fig5_scissors.pdf")
    if not os.path.isdir(os.path.dirname(out)):
        out = "fig5_scissors.pdf"
    fig.savefig(out); print(f"wrote {out}")


if __name__ == "__main__":
    main()
