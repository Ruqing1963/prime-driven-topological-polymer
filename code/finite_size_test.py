# -*- coding: utf-8 -*-
"""
Finite-size behaviour (paper Fig. 4 / Section IV D). True vs S1 at
N in {10000, 20000, 40000}, n=20 per arm; records Spearman rho and DeltaR.

At each fixed N both arms share identical annealing depth, so the True-S1 GAP is
the controlled quantity; the absolute rho/DeltaR trend across N is annealed
(non-equilibrium) and read only as a trend. Annealing steps grow with N.

Reduced force field (no Mersenne). Real-time CSV checkpointing with resume.

Usage:  python3 finite_size_test.py
        PDTP_MAXRUN=20 python3 finite_size_test.py   (do <=20 new runs, resume)
        SMOKE=1 python3 finite_size_test.py
NOTE: full run ~ several hours (40000 dominates); chunk with PDTP_MAXRUN.
Dependencies: numpy, scipy, matplotlib
"""
import os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats

SMOKE = os.environ.get("SMOKE", "0") == "1"
MAX_RUN = int(os.environ.get("PDTP_MAXRUN", 10**9))
BASE_SEED = 20260801
SCALES = {1_000: (60, 2), 2_000: (60, 2)} if SMOKE else {10_000: (2_000, 20), 20_000: (3_500, 20), 40_000: (5_000, 20)}

B0 = 1.0; K_BOND = 45.0; K_HINGE = 0.25; ALPHA = 0.15; K_MAX = 500.0
SIGMA = 1.0; EPS_LJ = 1.0; RC_LJ = 2.0 ** (1.0 / 6.0); FCAP_LJ = 30.0
MOBILITY = 1.0; K_CONF = 0.02; DT = 0.008; MAX_DISP = 0.25
KT_HI = 0.50; KT_LO = 0.003
MODELS = ["True", "S1"]; COLORS = {"True": "#c1272d", "S1": "#0868ac"}
CSV_PATH = "finite_size_results.csv"


def sieve_primes(n):
    ip = np.ones(n + 1, dtype=bool); ip[:2] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if ip[i]: ip[i * i::i] = False
    return ip

def gap_from_mask(is_anchor, N):
    anchors = np.where(is_anchor)[0]; g = np.zeros(N)
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b - a > 1: g[a + 1:b] = b - a
    g[0:anchors[0]] = anchors[0]; g[anchors[-1] + 1:N] = N - anchors[-1]; return g

def stiffness_from_gap(g, is_anchor):
    k = np.full(len(g), K_HINGE); comp = ~is_anchor
    k[comp] = np.minimum(K_HINGE * np.exp(ALPHA * g[comp]), K_MAX); return k

def mask_shuffle(primes, N, rng):
    gaps = np.diff(primes).copy(); rng.shuffle(gaps)
    newpos = primes[0] + np.concatenate(([0], np.cumsum(gaps)))
    mask = np.zeros(N, dtype=bool); mask[newpos] = True; return mask

def build_arm(model, primes, sieve_bool, N, rng):
    is_anchor = sieve_bool.copy() if model == "True" else mask_shuffle(primes, N, rng)
    g = gap_from_mask(is_anchor, N); return is_anchor, stiffness_from_gap(g, is_anchor)

def _scatter_add(F, idx, vals):
    N = F.shape[0]
    for c in range(3): F[:, c] += np.bincount(idx, weights=vals[:, c], minlength=N)

def compute_forces(r, kbend):
    N = r.shape[0]; F = np.zeros_like(r)
    d = r[1:] - r[:-1]; L = np.linalg.norm(d, axis=1) + 1e-12
    fb = (K_BOND * (L - B0) / L)[:, None] * d; F[0:N - 1] += fb; F[1:N] -= fb
    a = r[1:-1] - r[:-2]; b = r[2:] - r[1:-1]
    na = np.linalg.norm(a, axis=1) + 1e-12; nb = np.linalg.norm(b, axis=1) + 1e-12
    ah, bh = a / na[:, None], b / nb[:, None]
    cos = np.clip(np.sum(ah * bh, axis=1), -1.0, 1.0); kk = kbend[1:-1][:, None]
    dda = (bh - cos[:, None] * ah) / na[:, None]; ddb = (ah - cos[:, None] * bh) / nb[:, None]
    F[0:N - 2] += -kk * dda; F[2:N] += kk * ddb; F[1:N - 1] += kk * (dda - ddb)
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
    if K_CONF > 0.0: F += -K_CONF * (r - r.mean(axis=0))
    return F

def init_chain(N, rng, persistence=0.55):
    dirs = np.zeros((N, 3)); dv = rng.standard_normal(3); dv /= np.linalg.norm(dv); dirs[0] = dv
    for i in range(1, N):
        dv = persistence * dv + (1 - persistence) * rng.standard_normal(3)
        dv /= np.linalg.norm(dv) + 1e-12; dirs[i] = dv
    r = np.cumsum(dirs, axis=0) * B0; return r - r.mean(axis=0)

def run_langevin(r, kbend, n_steps, rng):
    for step in range(n_steps):
        frac = step / max(1, n_steps - 1); kT = KT_HI * (KT_LO / KT_HI) ** frac
        F = compute_forces(r, kbend)
        disp = MOBILITY * F * DT + np.sqrt(2.0 * MOBILITY * kT * DT) * rng.standard_normal(r.shape)
        dmag = np.linalg.norm(disp, axis=1); big = dmag > MAX_DISP
        if np.any(big): disp[big] *= (MAX_DISP / dmag[big])[:, None]
        r += disp
    return r

def observables(r, kbend):
    rad = np.linalg.norm(r - r.mean(axis=0), axis=1); rho, _ = stats.spearmanr(kbend, rad)
    hi = kbend > np.percentile(kbend, 80); lo = kbend < np.percentile(kbend, 20)
    return float(rho), float(rad[hi].mean() - rad[lo].mean())

def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            next(f, None)
            for line in f:
                p = line.strip().split(",")
                if len(p) == 5: done[(p[0], int(p[1]), int(p[2]))] = (float(p[3]), float(p[4]))
    return done

def append_row(path, model, N, run, rho, dR):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new: f.write("model,N,run,spearman_rho,dR\n")
        f.write(f"{model},{N},{run},{rho:.6f},{dR:.6f}\n")

def run_all():
    done = load_done(CSV_PATH); print(f"resume: {len(done)} runs done")
    executed = 0
    for N, (steps, n_ens) in SCALES.items():
        sieve_bool = sieve_primes(N - 1); primes = np.where(sieve_bool)[0]
        for mi, model in enumerate(MODELS):
            for k in range(n_ens):
                if (model, N, k) in done: continue
                if executed >= MAX_RUN: print(f"(reached PDTP_MAXRUN={MAX_RUN})"); return False
                si = list(SCALES.keys()).index(N)
                rng = np.random.default_rng(BASE_SEED + si * 1_000_000 + mi * 100_000 + k)
                is_anchor, kbend = build_arm(model, primes, sieve_bool, N, rng)
                r = init_chain(N, rng); t0 = time.time()
                r = run_langevin(r, kbend, steps, rng); rho, dR = observables(r, kbend)
                append_row(CSV_PATH, model, N, k, rho, dR); executed += 1
                print(f"[N={N:>6} {model} {k+1:02d}/{n_ens}] rho={rho:+.4f} dR={dR:+6.2f} ({time.time()-t0:.1f}s)")
    return True

def analyze_and_plot():
    done = load_done(CSV_PATH); Ns = sorted(SCALES.keys())
    agg = {m: {"rho": {}, "dR": {}} for m in MODELS}
    for (m, N, k), (rho, dR) in done.items():
        agg[m]["rho"].setdefault(N, []).append(rho); agg[m]["dR"].setdefault(N, []).append(dR)
    print("\n=== finite-size (True vs S1) ===")
    for N in Ns:
        for key, name in [("rho", "rho"), ("dR", "dR")]:
            t = np.array(agg["True"][key].get(N, [])); s = np.array(agg["S1"][key].get(N, []))
            if len(t) < 2 or len(s) < 2: continue
            p = float(stats.ttest_ind(t, s, equal_var=False).pvalue)
            print(f"  N={N:>6} {name}: True={t.mean():+.4f} S1={s.mean():+.4f} gap={t.mean()-s.mean():+.4f} p={p:.2e}")
    def ms(m, key):
        xs, ys, es = [], [], []
        for N in Ns:
            v = np.array(agg[m][key].get(N, []))
            if len(v) >= 2: xs.append(N); ys.append(v.mean()); es.append(v.std(ddof=1) / np.sqrt(len(v)))
        return xs, ys, es
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, key, ylab in [(ax[0], "rho", r"Spearman $\rho$"), (ax[1], "dR", r"$\Delta R$")]:
        for m in MODELS:
            xs, ys, es = ms(m, key)
            if xs: a.errorbar(xs, ys, yerr=es, marker="o", ms=7, lw=2, capsize=4, color=COLORS[m], label=m)
        a.axhline(0, color="gray", ls="--", lw=1); a.set_xscale("log"); a.set_xticks(Ns)
        a.set_xticklabels([str(N) for N in Ns]); a.set_xlabel("N"); a.set_ylabel(ylab)
        a.grid(alpha=0.3); a.legend()
    fig.tight_layout(); plt.savefig("finite_size_plot.pdf"); plt.close(fig)
    print("wrote finite_size_plot.pdf")

if __name__ == "__main__":
    t0 = time.time(); complete = run_all()
    if complete: analyze_and_plot(); print(f"done in {time.time()-t0:.1f}s -> {CSV_PATH}")
    else: print(f"[partial] saved {CSV_PATH}; rerun to continue. {time.time()-t0:.1f}s")
