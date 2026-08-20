# -*- coding: utf-8 -*-
"""
Scale-selective surrogate S4 -- Block Shuffle (paper Sec. V B, "A candidate
mechanism"). Delivers on the promise in Sec. 5.2 to localize the correlation
length responsible for the ordering signal, and to test the twin-prime
"topological trap" hypothesis directly.

--------------------------------------------------------------------------------
The mechanism under test
--------------------------------------------------------------------------------
Sec. 5.2 conjectures that dense clusters of small gaps -- twin primes (gap 2)
alone are ~14% of all gaps -- form *local* clumps of flexible hinges that trap
neighboring rigid rods in the interior, suppressing the stiffness-radius rank
correlation rho below a fully scattered (shuffled) arrangement. If that is the
mechanism, the signal lives in a *local* correlation length: preserving gap
order over contiguous windows of size L_block should recover the true-sequence
suppression, while destroying it (small L_block) should collapse rho back onto
the ordinary shuffle S1.

--------------------------------------------------------------------------------
The S4 surrogate (a controlled interpolation between S1 and True)
--------------------------------------------------------------------------------
Take the ordered prime-gap sequence g = diff(primes). Cut it into CONTIGUOUS
blocks of length L_block. Keep each block's internal gap order INTACT (local
twin-prime clusters survive), then apply a random permutation to the ORDER OF
THE BLOCKS (large-scale arithmetic arrangement destroyed). Rebuild the anchor
mask from the reassembled gaps.

  L_block = 1        -> every gap is its own block  -> full shuffle == S1
  L_block >= len(g)  -> a single block, unpermuted  -> True prime sequence
  1 < L_block < len  -> scale-selective interpolation

S4 preserves the gap multiset EXACTLY (same marginal stiffness distribution as
True and S1) and the anchor count exactly; only the *range* of preserved order
changes. It is therefore a legitimate null in the same family as S1, differing
only in the correlation length it retains. Sweeping L_block maps out the scale
at which core-shell ordering is recovered / collapses.

--------------------------------------------------------------------------------
Design / comparability
--------------------------------------------------------------------------------
Same reduced force field, annealing protocol, N, and step count as the base
True-vs-S1 run (decomposition_test.py / true_vs_s1_results.csv), so S4 rho is
directly comparable to the imported True and S1 ensembles. Seeds use the same
BASE_SEED=20260715 convention with disjoint model indices (True=0, S1=1, then
one index per L_block) so no seed stream collides. True and S1 rows are imported
from true_vs_s1_results.csv when present; only the S4 arms are simulated.

As in the paper, True/S1/S4 are independent ensembles compared by a shift in the
ensemble mean (Welch t-test, Cohen d); absolute rho depends on the (annealed,
non-equilibrium) protocol, so only the fixed-N gaps between arms are controlled.

Usage:  python3 block_shuffle_test.py
        SMOKE=1 python3 block_shuffle_test.py                 # tiny sanity run
        LBLOCKS="10,50,100,500" python3 block_shuffle_test.py # override scales
        PDTP_MAXRUN=20 python3 block_shuffle_test.py          # cap new sims, resume
Dependencies: numpy, scipy, matplotlib
"""
import os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats

# ------------------------------------------------------------------ config ----
SMOKE = os.environ.get("SMOKE", "0") == "1"
N          = int(os.environ.get("NN",     10_000 if not SMOKE else 1_000))
N_STEPS    = int(os.environ.get("NSTEPS",  2_000 if not SMOKE else 60))
N_ENSEMBLE = int(os.environ.get("NENS",       40 if not SMOKE else 2))
MAX_RUN    = int(os.environ.get("PDTP_MAXRUN", 10**9))
BASE_SEED  = 20260715            # same seeding convention as decomposition_test

# scale ladder (block lengths, in gaps). Default per the Sec. 5.2 proposal.
_default_L = "10,50,100,500" if not SMOKE else "1,4,16"
L_BLOCKS   = [int(x) for x in os.environ.get("LBLOCKS", _default_L).split(",")]

# force field (identical to decomposition_test.py / finite_size_test.py)
B0 = 1.0; K_BOND = 45.0; K_HINGE = 0.25; ALPHA = 0.15; K_MAX = 500.0
SIGMA = 1.0; EPS_LJ = 1.0; RC_LJ = 2.0 ** (1.0 / 6.0); FCAP_LJ = 30.0
MOBILITY = 1.0; K_CONF = 0.02; DT = 0.008; MAX_DISP = 0.25
KT_HI = 0.50; KT_LO = 0.003

CSV_PATH  = "block_shuffle_results.csv"
PRIOR_CSV = "true_vs_s1_results.csv"      # source of imported True / S1 rows

REF = {"True": "#c1272d", "S1": "#0868ac"}
# each L_block gets a distinct model tag "S4_L<block>" and a model index >= 10
def s4_tag(L):  return f"S4_L{L}"
def s4_mi(L):   return 10 + L_BLOCKS.index(L)     # disjoint from True(0)/S1(1)/S3(2)


# ----------------------------------------------------------- arithmetic -------
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

def mask_block_shuffle(primes, N, L_block, rng):
    """Block-preserving permutation of the prime-gap sequence.

    Contiguous blocks of length L_block keep their internal gap order; the blocks
    themselves are globally shuffled. Preserves the gap multiset and anchor count
    exactly. L_block=1 -> full shuffle (S1); L_block>=len(gaps) -> True.
    """
    gaps = np.diff(primes)
    n = len(gaps)
    n_blocks = int(np.ceil(n / L_block))
    order = rng.permutation(n_blocks)
    blocks = [gaps[b * L_block:(b + 1) * L_block] for b in range(n_blocks)]
    new_gaps = np.concatenate([blocks[b] for b in order])
    newpos = primes[0] + np.concatenate(([0], np.cumsum(new_gaps)))
    mask = np.zeros(N, dtype=bool); mask[newpos] = True
    return mask

def build_s4_arm(primes, N, L_block, rng):
    is_anchor = mask_block_shuffle(primes, N, L_block, rng)
    g = gap_from_mask(is_anchor, N)
    return is_anchor, stiffness_from_gap(g, is_anchor)


# --------------------------------------------------------------- physics ------
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
    rad = np.linalg.norm(r - r.mean(axis=0), axis=1)
    rho, _ = stats.spearmanr(kbend, rad); return float(rho)


# ----------------------------------------------------------- checkpointing ----
def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            next(f, None)
            for line in f:
                p = line.strip().split(",")
                if len(p) == 3:
                    done[(p[0], int(p[1]))] = float(p[2])
    return done

def append_row(path, model, run, rho):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new: f.write("model,run,spearman_rho\n")
        f.write(f"{model},{run},{rho:.6f}\n")

def import_prior():
    """Bring True and S1 rows over from true_vs_s1_results.csv (same seeding)."""
    if not os.path.exists(PRIOR_CSV):
        print(f"[warn] {PRIOR_CSV} not found -- True/S1 reference bands will be "
              f"absent unless you generate them first (decomposition/finite_size).")
        return
    prior = load_done(PRIOR_CSV); have = load_done(CSV_PATH); added = 0
    for (m, k), rho in sorted(prior.items()):
        if m in ("True", "S1") and 0 <= k < N_ENSEMBLE and (m, k) not in have:
            append_row(CSV_PATH, m, k, rho); added += 1
    if added: print(f"imported {added} True/S1 rows from {PRIOR_CSV}")


# --------------------------------------------------------------- driver -------
def run_ensemble():
    sieve_bool = sieve_primes(N - 1); primes = np.where(sieve_bool)[0]
    import_prior()
    done = load_done(CSV_PATH)
    tags = ["True", "S1"] + [s4_tag(L) for L in L_BLOCKS]
    results = {t: [None] * N_ENSEMBLE for t in tags}
    for (m, k), rho in done.items():
        if m in results and 0 <= k < N_ENSEMBLE: results[m][k] = rho
    n_have = sum(v is not None for t in tags for v in results[t])
    print(f"resume: {n_have}/{len(tags)*N_ENSEMBLE} cells filled "
          f"(N={N}, steps={N_STEPS}, n={N_ENSEMBLE}, L={L_BLOCKS})")

    executed = 0
    for L in L_BLOCKS:                       # only S4 arms are simulated here
        tag, mi = s4_tag(L), s4_mi(L)
        for k in range(N_ENSEMBLE):
            if results[tag][k] is not None: continue
            if executed >= MAX_RUN:
                print(f"(reached PDTP_MAXRUN={MAX_RUN}); rerun to continue")
                return results, tags, False
            rng = np.random.default_rng(BASE_SEED + mi * 100_000 + k)
            _, kbend = build_s4_arm(primes, N, L, rng)
            r = init_chain(N, rng); t0 = time.time()
            r = run_langevin(r, kbend, N_STEPS, rng); rho = core_shell_spearman(r, kbend)
            results[tag][k] = rho; append_row(CSV_PATH, tag, k, rho); executed += 1
            print(f"[{tag:>8} {k+1:02d}/{N_ENSEMBLE}] rho={rho:+.4f} ({time.time()-t0:.1f}s)")
    return results, tags, True


# --------------------------------------------------------------- analysis -----
def pair_stats(a, b):
    """(mean gap a-b, Welch p, Cohen d) with pooled-sd normalization."""
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2) + 1e-12
    return a.mean() - b.mean(), p, (a.mean() - b.mean()) / pooled

def analyze_and_plot(results):
    arr = {t: np.array([x for x in results[t] if x is not None]) for t in results}
    have_ref = len(arr.get("True", [])) >= 2 and len(arr.get("S1", [])) >= 2

    print("\n=== S4 block-shuffle: critical-scale sweep (Spearman rho) ===")
    if have_ref:
        T, S1 = arr["True"], arr["S1"]
        span = S1.mean() - T.mean()   # S1 - True = full ordering suppression (>0)
        print(f"  reference  True = {T.mean():+.4f} +/- {T.std(ddof=1)/len(T)**.5:.4f}"
              f"   S1 = {S1.mean():+.4f} +/- {S1.std(ddof=1)/len(S1)**.5:.4f}"
              f"   (S1-True span = {span:+.4f})")
        print(f"  {'L_block':>8} {'rho_S4':>16} {'gap vs S1':>12} {'gap vs True':>12}"
              f" {'recovery':>9}")
        for L in L_BLOCKS:
            v = arr.get(s4_tag(L), np.array([]))
            if len(v) < 2:
                print(f"  {L:>8}  (insufficient runs)"); continue
            g1, p1, d1 = pair_stats(v, S1)     # negative g1 = suppressed toward True
            gT, pT, dT = pair_stats(v, T)
            rec = (S1.mean() - v.mean()) / span if abs(span) > 1e-9 else float("nan")
            m = f"{v.mean():+.4f}+/-{v.std(ddof=1)/len(v)**.5:.4f}"
            sig1 = "SIG" if p1 < 0.05 else "n.s."
            print(f"  {L:>8} {m:>16} {g1:+7.4f}({sig1}) {gT:+11.4f} {rec:>8.2f}")
        print("\n  recovery = (rho_S1 - rho_S4)/(rho_S1 - rho_True):"
              " 0 = shuffle-like (order collapsed), 1 = true-like (order restored).")
    else:
        print("  [no True/S1 reference in CSV; printing S4 means only]")
        for L in L_BLOCKS:
            v = arr.get(s4_tag(L), np.array([]))
            if len(v) >= 2:
                print(f"  L={L:>6}: rho_S4 = {v.mean():+.4f} +/- "
                      f"{v.std(ddof=1)/len(v)**.5:.4f}  (n={len(v)})")

    # ---- figure: rho vs L_block, with True/S1 reference bands + recovery axis
    Ls, mu, se = [], [], []
    for L in L_BLOCKS:
        v = arr.get(s4_tag(L), np.array([]))
        if len(v) >= 2:
            Ls.append(L); mu.append(v.mean()); se.append(v.std(ddof=1) / np.sqrt(len(v)))
    if not Ls:
        print("no S4 points to plot yet."); return
    Ls, mu, se = np.array(Ls), np.array(mu), np.array(se)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.errorbar(Ls, mu, yerr=se, marker="o", ms=8, lw=2, capsize=4,
                color="#6a3d9a", zorder=4, label=r"S4 block shuffle")
    if have_ref:
        for name in ("True", "S1"):
            m = arr[name].mean(); s = arr[name].std(ddof=1) / np.sqrt(len(arr[name]))
            ax.axhline(m, color=REF[name], ls="--", lw=1.6,
                       label=fr"{name} (L$\to${'N' if name=='True' else '1'})")
            ax.fill_between([Ls.min() * 0.7, Ls.max() * 1.4], m - s, m + s,
                            color=REF[name], alpha=0.12, zorder=0)
    ax.set_xscale("log"); ax.set_xlabel(r"block length $L_{\mathrm{block}}$ (gaps)")
    ax.set_ylabel(r"Spearman $\rho$ (stiffness vs radius)")
    ax.set_xlim(Ls.min() * 0.7, Ls.max() * 1.4)
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="best", fontsize=9)
    ax.set_title("S4: recovery of core--shell ordering vs preserved correlation length")
    fig.tight_layout(); plt.savefig("block_shuffle_plot.pdf"); plt.close(fig)
    print("wrote block_shuffle_plot.pdf")


if __name__ == "__main__":
    t0 = time.time()
    results, tags, complete = run_ensemble()
    if complete:
        analyze_and_plot(results)
        print(f"done in {time.time()-t0:.1f}s -> {CSV_PATH}")
    else:
        print(f"[partial] saved {CSV_PATH}; rerun to continue. {time.time()-t0:.1f}s")
