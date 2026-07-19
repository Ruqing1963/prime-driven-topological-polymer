# -*- coding: utf-8 -*-
"""
Decisive test (paper Fig. 2 / Section IV B): is the ordering effect at N=40000
real, or a power/annealing artifact? Runs True vs S1 at N=40000 with deep
annealing (8000 steps, i.e. 0.20 steps/bead, matched to N=10000) and n=40 per
arm. Records Spearman rho(stiffness, radius) and DeltaR (p80-p20 radii).

Reduced force field (no Mersenne): bond + gap-driven bending + WCA + weak
confinement. Real-time CSV checkpointing with resume.

Usage:  python3 decisive_40k_test.py
        PDTP_MAXRUN=10 python3 decisive_40k_test.py   (do <=10 new runs, resume)
        SMOKE=1 python3 decisive_40k_test.py           (tiny sanity check)
NOTE: full run is ~80 trajectories x ~5-6 min each ~ 7-8 h; chunk with PDTP_MAXRUN.
Dependencies: numpy, scipy, matplotlib
"""
import os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats

SMOKE   = os.environ.get("SMOKE", "0") == "1"
MAX_RUN = int(os.environ.get("PDTP_MAXRUN", 10**9))
BASE_SEED = 20260901
N          = 40_000 if not SMOKE else 2_000
N_STEPS    = 8_000  if not SMOKE else 100     # 8000/40000 = 0.20 steps/bead
N_ENSEMBLE = 40     if not SMOKE else 2

B0 = 1.0; K_BOND = 45.0; K_HINGE = 0.25; ALPHA = 0.15; K_MAX = 500.0
SIGMA = 1.0; EPS_LJ = 1.0; RC_LJ = 2.0 ** (1.0 / 6.0); FCAP_LJ = 30.0
MOBILITY = 1.0; K_CONF = 0.02; DT = 0.008; MAX_DISP = 0.25
KT_HI = 0.50; KT_LO = 0.003
MODELS = ["True", "S1"]; COLORS = {"True": "#c1272d", "S1": "#0868ac"}
CSV_PATH = "decisive_40k_results.csv"


def sieve_primes(n):
    ip = np.ones(n + 1, dtype=bool); ip[:2] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if ip[i]: ip[i * i::i] = False
    return ip

def gap_from_mask(is_anchor, N):
    anchors = np.where(is_anchor)[0]; g = np.zeros(N)
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b - a > 1: g[a + 1:b] = b - a
    g[0:anchors[0]] = anchors[0]; g[anchors[-1] + 1:N] = N - anchors[-1]
    return g

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
    rad = np.linalg.norm(r - r.mean(axis=0), axis=1)
    rho, _ = stats.spearmanr(kbend, rad)
    hi = kbend > np.percentile(kbend, 80); lo = kbend < np.percentile(kbend, 20)
    return float(rho), float(rad[hi].mean() - rad[lo].mean())

def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            next(f, None)
            for line in f:
                p = line.strip().split(",")
                if len(p) == 4: done[(p[0], int(p[1]))] = (float(p[2]), float(p[3]))
    return done

def append_row(path, model, run, rho, dR):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new: f.write("model,run,spearman_rho,dR\n")
        f.write(f"{model},{run},{rho:.6f},{dR:.6f}\n")

def run_ensemble():
    sieve_bool = sieve_primes(N - 1); primes = np.where(sieve_bool)[0]
    done = load_done(CSV_PATH)
    results = {m: {"rho": [None] * N_ENSEMBLE, "dR": [None] * N_ENSEMBLE} for m in MODELS}
    for (m, k), (rho, dR) in done.items():
        if m in results and 0 <= k < N_ENSEMBLE: results[m]["rho"][k] = rho; results[m]["dR"][k] = dR
    print(f"resume: {len(done)}/{len(MODELS)*N_ENSEMBLE} done  (N={N}, {N_STEPS} steps, n={N_ENSEMBLE}/arm)")
    executed = 0
    for mi, model in enumerate(MODELS):
        for k in range(N_ENSEMBLE):
            if results[model]["rho"][k] is not None: continue
            if executed >= MAX_RUN:
                print(f"(reached PDTP_MAXRUN={MAX_RUN}; rerun to continue)"); return results, False
            rng = np.random.default_rng(BASE_SEED + mi * 100_000 + k)
            is_anchor, kbend = build_arm(model, primes, sieve_bool, N, rng)
            r = init_chain(N, rng); t0 = time.time()
            r = run_langevin(r, kbend, N_STEPS, rng)
            rho, dR = observables(r, kbend)
            results[model]["rho"][k] = rho; results[model]["dR"][k] = dR
            append_row(CSV_PATH, model, k, rho, dR); executed += 1
            print(f"[{model} {k+1:02d}/{N_ENSEMBLE}] rho={rho:+.4f} dR={dR:+6.2f} ({time.time()-t0:.1f}s)")
    complete = all(results[m]["rho"][k] is not None for m in MODELS for k in range(N_ENSEMBLE))
    return results, complete

def pair_stats(a, b):
    a, b = np.array(a), np.array(b)
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    try: mwu = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError: mwu = float("nan")
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2) + 1e-12
    return a.mean() - b.mean(), p, mwu, (a.mean() - b.mean()) / pooled

def analyze_and_plot(results):
    arms = {m: {k: np.array([x for x in results[m][k] if x is not None]) for k in ("rho", "dR")} for m in MODELS}
    if any(len(arms[m]["rho"]) < 2 for m in MODELS): print("insufficient data"); return
    print("\n=== Decisive 40k (deep anneal + n=40) ===")
    for key, name in [("rho", "Spearman rho"), ("dR", "DeltaR")]:
        dm, p, mwu, d = pair_stats(arms["True"][key], arms["S1"][key])
        print(f" {name}: True={arms['True'][key].mean():+.4f} S1={arms['S1'][key].mean():+.4f} "
              f"gap={dm:+.4f} Welch p={p:.2e} MWU={mwu:.2e} d={d:+.2f} {'SIG' if p<0.05 else 'n.s.'}")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    for a, key, ylab in [(ax[0], "rho", r"Spearman $\rho$"), (ax[1], "dR", r"$\Delta R$")]:
        data = [arms["True"][key], arms["S1"][key]]
        bp = a.boxplot(data, positions=[1, 2], widths=0.5, patch_artist=True, showfliers=False,
                       medianprops=dict(color="k", lw=2))
        for patch, m in zip(bp["boxes"], MODELS): patch.set_facecolor(COLORS[m]); patch.set_alpha(0.35)
        rng = np.random.default_rng(0)
        for pos, m in zip([1, 2], MODELS):
            y = arms[m][key]; a.scatter(pos + (rng.random(len(y)) - 0.5) * 0.16, y, s=30,
                                        color=COLORS[m], edgecolor="k", lw=0.5, zorder=3)
        a.axhline(0, color="gray", ls="--", lw=1); a.set_xticks([1, 2])
        a.set_xticklabels(["True primes", "S1 shuffle"]); a.set_ylabel(ylab); a.grid(axis="y", alpha=0.3)
    fig.tight_layout(); plt.savefig("decisive_40k_plot.pdf"); plt.close(fig)
    print("wrote decisive_40k_plot.pdf")

if __name__ == "__main__":
    t0 = time.time(); results, complete = run_ensemble()
    if complete:
        analyze_and_plot(results); print(f"done in {time.time()-t0:.1f}s -> {CSV_PATH}")
    else:
        print(f"[partial] saved {CSV_PATH}; rerun to continue. {time.time()-t0:.1f}s")
