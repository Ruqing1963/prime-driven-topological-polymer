# -*- coding: utf-8 -*-
"""
Generate the representative 3D conformation (paper Fig. 1) with the reduced PDTP
force field: bond + gap-driven bending [Eq. (1)] + WCA self-avoidance + weak
confinement. No Mersenne term. Single annealing trajectory at N = 40000.

Outputs:
  conformation_coords.npy   final coordinates (N,3)
  conformation_kbend.npy    per-bead stiffness k_i
  conformation.pdf          3D scatter colored by log10(k_i)

Usage:  python3 conformation.py            (N=40000, ~2.5-3 min)
        SMOKE=1 python3 conformation.py     (tiny, sanity check)
Dependencies: numpy, scipy, matplotlib
"""
import os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from scipy.spatial import cKDTree

SMOKE   = os.environ.get("SMOKE", "0") == "1"
N       = 40_000 if not SMOKE else 3_000
N_STEPS = 3_800  if not SMOKE else 300
SEED    = 42

B0 = 1.0; K_BOND = 45.0; K_HINGE = 0.25; ALPHA = 0.15; K_MAX = 500.0
SIGMA = 1.0; EPS_LJ = 1.0; RC_LJ = 2.0 ** (1.0 / 6.0); FCAP_LJ = 30.0
MOBILITY = 1.0; K_CONF = 0.02; DT = 0.008; MAX_DISP = 0.25
KT_HI = 0.50; KT_LO = 0.003
RNG = np.random.default_rng(SEED)


def sieve_primes(n):
    ip = np.ones(n + 1, dtype=bool); ip[:2] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if ip[i]: ip[i * i::i] = False
    return ip

def prime_gap_array(is_prime, N):
    primes = np.where(is_prime)[0]; g = np.zeros(N)
    for a, b in zip(primes[:-1], primes[1:]):
        if b - a > 1: g[a + 1:b] = b - a
    g[0:primes[0]] = primes[0]; g[primes[-1] + 1:N] = N - primes[-1]
    return g

def build_stiffness(N):
    is_prime = sieve_primes(N - 1); gap = prime_gap_array(is_prime, N)
    k = np.full(N, K_HINGE); comp = ~is_prime
    k[comp] = np.minimum(K_HINGE * np.exp(ALPHA * gap[comp]), K_MAX)
    return k, gap, is_prime

def _scatter_add(F, idx, vals):
    N = F.shape[0]
    for c in range(3): F[:, c] += np.bincount(idx, weights=vals[:, c], minlength=N)

def compute_forces(r, kbend):
    N = r.shape[0]; F = np.zeros_like(r)
    # bonds
    d = r[1:] - r[:-1]; L = np.linalg.norm(d, axis=1) + 1e-12
    fb = (K_BOND * (L - B0) / L)[:, None] * d; F[0:N - 1] += fb; F[1:N] -= fb
    # gap-driven bending  E_i = k_i (1 - cos theta_i)
    a = r[1:-1] - r[:-2]; b = r[2:] - r[1:-1]
    na = np.linalg.norm(a, axis=1) + 1e-12; nb = np.linalg.norm(b, axis=1) + 1e-12
    ah, bh = a / na[:, None], b / nb[:, None]
    cos = np.clip(np.sum(ah * bh, axis=1), -1.0, 1.0); kk = kbend[1:-1][:, None]
    dda = (bh - cos[:, None] * ah) / na[:, None]; ddb = (ah - cos[:, None] * bh) / nb[:, None]
    F[0:N - 2] += -kk * dda; F[2:N] += kk * ddb; F[1:N - 1] += kk * (dda - ddb)
    # WCA repulsion
    tree = cKDTree(r); pairs = tree.query_pairs(RC_LJ * SIGMA, output_type="ndarray")
    if len(pairs) > 0:
        i, j = pairs[:, 0], pairs[:, 1]; keep = np.abs(i - j) > 1; i, j = i[keep], j[keep]
        for s in range(0, len(i), 4_000_000):
            ii, jj = i[s:s + 4_000_000], j[s:s + 4_000_000]
            rij = r[ii] - r[jj]; dist = np.linalg.norm(rij, axis=1) + 1e-12
            sr6 = (SIGMA / dist) ** 6
            fmag = np.clip(24 * EPS_LJ * (2 * sr6 ** 2 - sr6) / dist, 0.0, FCAP_LJ)
            fvec = (fmag / dist)[:, None] * rij
            _scatter_add(F, ii, fvec); _scatter_add(F, jj, -fvec)
    # weak confinement
    if K_CONF > 0.0: F += -K_CONF * (r - r.mean(axis=0))
    return F

def radius_of_gyration(r):
    c = r.mean(axis=0); return float(np.sqrt(np.mean(np.sum((r - c) ** 2, axis=1))))

def init_chain(N, persistence=0.55):
    dirs = np.zeros((N, 3)); dv = RNG.standard_normal(3); dv /= np.linalg.norm(dv); dirs[0] = dv
    for i in range(1, N):
        dv = persistence * dv + (1 - persistence) * RNG.standard_normal(3)
        dv /= np.linalg.norm(dv) + 1e-12; dirs[i] = dv
    r = np.cumsum(dirs, axis=0) * B0; return r - r.mean(axis=0)

def run_langevin(r, kbend, n_steps):
    for step in range(n_steps):
        frac = step / max(1, n_steps - 1)
        kT = KT_HI * (KT_LO / KT_HI) ** frac
        F = compute_forces(r, kbend)
        disp = MOBILITY * F * DT + np.sqrt(2.0 * MOBILITY * kT * DT) * RNG.standard_normal(r.shape)
        dmag = np.linalg.norm(disp, axis=1); big = dmag > MAX_DISP
        if np.any(big): disp[big] *= (MAX_DISP / dmag[big])[:, None]
        r += disp
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            print(f"  step {step:>5}/{n_steps}  kT={kT:.4f}  R_g={radius_of_gyration(r):.2f}")
    return r

def plot_conformation(r, kbend, fname):
    fig = plt.figure(figsize=(9, 8)); ax = fig.add_subplot(111, projection="3d")
    c = np.log10(np.clip(kbend, 1e-3, None))
    ax.plot(r[:, 0], r[:, 1], r[:, 2], color="0.8", lw=0.3, alpha=0.5, zorder=1)
    sc = ax.scatter(r[:, 0], r[:, 1], r[:, 2], c=c, cmap="coolwarm", s=4, alpha=0.75, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(r"$\log_{10}\,k_i$   (low = flexible core,  high = rigid rod)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ext = np.array([r[:, k].max() - r[:, k].min() for k in range(3)])
    try: ax.set_box_aspect(ext)
    except Exception: pass
    ax.view_init(elev=18, azim=-60); fig.tight_layout()
    fig.savefig(fname); plt.close(fig)

def main():
    t0 = time.time(); kbend, gap, is_prime = build_stiffness(N)
    print(f"N={N}  primes={int(is_prime.sum())}  g_max={int(gap.max())}  "
          f"k in [{kbend.min():.2f},{kbend.max():.1f}] ({kbend.max()/kbend.min():.0f}x)")
    r = init_chain(N); r = run_langevin(r, kbend, N_STEPS)
    print(f"final R_g = {radius_of_gyration(r):.3f} | {time.time()-t0:.1f}s")
    np.save("conformation_coords.npy", r); np.save("conformation_kbend.npy", kbend)
    plot_conformation(r, kbend, "conformation.pdf")
    print("wrote conformation.pdf, conformation_coords.npy, conformation_kbend.npy")

if __name__ == "__main__":
    main()
