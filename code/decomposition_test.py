# -*- coding: utf-8 -*-
"""
Ordering vs marginal-distribution decomposition (paper Fig. 3 / Section IV C).
Three arms at N=10000, n=40, observable = Spearman rho(stiffness, radius):

  True : real prime-gap sequence.
  S1   : shuffle -- exact gap multiset, order randomized (isolates ORDERING).
  S3   : Cramer -- pseudo-primes placed at density 1/ln(i) (random marginal).

  True vs S1 = ordering effect;  S1 vs S3 = marginal-shape effect;
  True vs S3 = combined.

Reduced force field (no Mersenne). Real-time CSV checkpointing with resume.
If a prior 'true_vs_s1_results.csv' (same seeding: BASE_SEED=20260715, mi
True=0/S1=1) is present, its True/S1 rows are imported and only S3 is computed.

Usage:  python3 decomposition_test.py
        SMOKE=1 python3 decomposition_test.py
Dependencies: numpy, scipy, matplotlib
"""
import os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats

SMOKE = os.environ.get("SMOKE", "0") == "1"
N          = int(os.environ.get("NN",     10_000 if not SMOKE else 1_000))
N_STEPS    = int(os.environ.get("NSTEPS",  2_000 if not SMOKE else 60))
N_ENSEMBLE = int(os.environ.get("NENS",       40 if not SMOKE else 2))
BASE_SEED  = 20260715

B0 = 1.0; K_BOND = 45.0; K_HINGE = 0.25; ALPHA = 0.15; K_MAX = 500.0
SIGMA = 1.0; EPS_LJ = 1.0; RC_LJ = 2.0 ** (1.0 / 6.0); FCAP_LJ = 30.0
MOBILITY = 1.0; K_CONF = 0.02; DT = 0.008; MAX_DISP = 0.25
KT_HI = 0.50; KT_LO = 0.003
MODELS = ["True", "S1", "S3"]; COLORS = {"True": "#c1272d", "S1": "#0868ac", "S3": "#238b45"}
CSV_PATH = "decomposition_results.csv"; PRIOR_CSV = "true_vs_s1_results.csv"


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

def mask_cramer(N, rng):
    i = np.arange(N, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"): p = 1.0 / np.log(i)
    p[:2] = 0.0; p = np.clip(p, 0.0, 1.0)
    mask = rng.random(N) < p
    if mask.sum() < 2: mask[[2, N - 1]] = True
    return mask

def build_arm(model, primes, sieve_bool, N, rng):
    if model == "True": is_anchor = sieve_bool.copy()
    elif model == "S1": is_anchor = mask_shuffle(primes, N, rng)
    else: is_anchor = mask_cramer(N, rng)
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

def core_shell_spearman(r, kbend):
    rad = np.linalg.norm(r - r.mean(axis=0), axis=1); rho, _ = stats.spearmanr(kbend, rad); return float(rho)

def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            next(f, None)
            for line in f:
                p = line.strip().split(",")
                if len(p) == 3: done[(p[0], int(p[1]))] = float(p[2])
    return done

def append_row(path, model, run, rho):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new: f.write("model,run,spearman_rho\n")
        f.write(f"{model},{run},{rho:.6f}\n")

def import_prior():
    if not os.path.exists(PRIOR_CSV): return
    prior = load_done(PRIOR_CSV); have = load_done(CSV_PATH); added = 0
    for (m, k), rho in sorted(prior.items()):
        if m in MODELS and 0 <= k < N_ENSEMBLE and (m, k) not in have:
            append_row(CSV_PATH, m, k, rho); added += 1
    print(f"imported {added} True/S1 rows from {PRIOR_CSV}")

def run_ensemble():
    sieve_bool = sieve_primes(N - 1); primes = np.where(sieve_bool)[0]
    import_prior(); done = load_done(CSV_PATH)
    results = {m: [None] * N_ENSEMBLE for m in MODELS}
    for (m, k), rho in done.items():
        if m in results and 0 <= k < N_ENSEMBLE: results[m][k] = rho
    print(f"resume: {len(done)}/{len(MODELS)*N_ENSEMBLE} done")
    for mi, model in enumerate(MODELS):
        for k in range(N_ENSEMBLE):
            if results[model][k] is not None: continue
            rng = np.random.default_rng(BASE_SEED + mi * 100_000 + k)
            is_anchor, kbend = build_arm(model, primes, sieve_bool, N, rng)
            r = init_chain(N, rng); t0 = time.time()
            r = run_langevin(r, kbend, N_STEPS, rng); rho = core_shell_spearman(r, kbend)
            results[model][k] = rho; append_row(CSV_PATH, model, k, rho)
            print(f"[{model} {k+1:02d}/{N_ENSEMBLE}] rho={rho:+.4f} ({time.time()-t0:.1f}s)")
    return {m: np.array([x for x in results[m] if x is not None]) for m in MODELS}

def pair_stats(a, b):
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2) + 1e-12
    return a.mean() - b.mean(), p, (a.mean() - b.mean()) / pooled

def analyze_and_plot(res):
    T, S1, S3 = res["True"], res["S1"], res["S3"]
    print("\n=== decomposition (Spearman rho) ===")
    for a, b, name in [("True", "S1", "ordering"), ("S1", "S3", "marginal-shape"), ("True", "S3", "combined")]:
        dm, p, d = pair_stats(res[a], res[b])
        print(f" {a} vs {b} [{name}]: gap={dm:+.4f} p={p:.2e} d={d:+.2f} {'SIG' if p<0.05 else 'n.s.'}")
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    bp = ax.boxplot([T, S1, S3], positions=[1, 2, 3], widths=0.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="k", lw=2))
    for patch, m in zip(bp["boxes"], MODELS): patch.set_facecolor(COLORS[m]); patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for pos, m in zip([1, 2, 3], MODELS):
        y = res[m]; ax.scatter(pos + (rng.random(len(y)) - 0.5) * 0.18, y, s=30,
                               color=COLORS[m], edgecolor="k", lw=0.5, zorder=3)
    ax.axhline(0, color="gray", ls="--", lw=1); ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["True primes\n(ordered)", "S1 shuffle\n(real marginal)", "S3 Cramer\n(random marginal)"])
    ax.set_ylabel(r"Spearman $\rho$ (stiffness vs radius)"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); plt.savefig("decomposition_plot.pdf"); plt.close(fig)
    print("wrote decomposition_plot.pdf")

if __name__ == "__main__":
    t0 = time.time(); res = run_ensemble()
    if all(len(res[m]) >= 2 for m in MODELS): analyze_and_plot(res)
    print(f"done in {time.time()-t0:.1f}s -> {CSV_PATH}")
