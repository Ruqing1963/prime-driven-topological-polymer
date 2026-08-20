# -*- coding: utf-8 -*-
"""
Scale-selective surrogate S5 -- Local Scramble (converse of S4). Closes the
"scissors" test proposed for paper Sec. 5.2: together with S4 it localizes the
correlation length responsible for the core-shell ordering signal.

--------------------------------------------------------------------------------
The two blades of the scissors
--------------------------------------------------------------------------------
Both surrogates cut the ordered prime-gap sequence g = diff(primes) into
CONTIGUOUS blocks of length L_block. They then destroy order at complementary
scales, each preserving the gap multiset (hence the marginal stiffness
distribution) and the anchor count EXACTLY:

  S4 (block shuffle) : keep order WITHIN blocks, shuffle the ORDER OF blocks.
                       Destroys long-range arrangement, keeps local order.
                       L=1 -> full shuffle (S1);  L>=n -> True.
                       [result: rho stays pinned at the shuffle S1 level at every
                        tested L up to 500 -- the signal is NOT recovered.]

  S5 (local scramble): keep the blocks in their TRUE GLOBAL ORDER, scramble the
                       gaps WITHIN each block. Destroys local order, keeps the
                       long-range arrangement.
                       L=1 -> True;  L>=n -> full shuffle (S1).

Reading the scissors:
  * If the signal is carried by LOCAL structure, S5 collapses to S1 quickly
    (even small blocks destroy it) while S4 stays near True.
  * If the signal is carried by LONG-RANGE / GLOBAL arithmetic order, S5 should
    HUG True across a wide range of L (local scrambling is harmless) while S4
    sits at S1 -- which is what the S4 run already showed. S5 is the decisive
    second blade.

--------------------------------------------------------------------------------
--num-swaps : titrating local disorder without the block-count confound
--------------------------------------------------------------------------------
S4 at large L suffered a combinatorial-starvation artifact: with only 3 blocks
there are just 3! = 6 arrangements, so its ensemble variance exploded. S5 avoids
this by titrating disorder with a CONTINUOUS dial rather than sampling a shrinking
permutation space: --num-swaps k applies k random adjacent-pair transpositions
*within each block* (per block), then rebuilds the mask.

  --num-swaps 0     -> no scrambling                 -> True   (any L)
  --num-swaps k     -> k within-block transpositions -> partial local disorder
  --num-swaps full  -> complete within-block shuffle -> pure S5(L)

Saturation to the full-shuffle limit occurs near k ~ 0.5 * L * ln L swaps per
block. Setting L = N (one block = the whole sequence) turns --num-swaps into a
smooth, confound-free walk from True (k=0) to S1 (k large): this is the default
when --num-swaps is given without --l-block.

--------------------------------------------------------------------------------
Consistency (directly comparable to True / S1 / S4)
--------------------------------------------------------------------------------
Identical reduced force field, annealing protocol, N=10000, N_STEPS=2000, and
BASE_SEED=20260715 convention as the base True-vs-S1 run and the S4 test. Model
seed indices live in a disjoint band (>=1000) so no seed stream collides with
True(0)/S1(1)/S3(2)/S4(10-13). True and S1 rows are auto-imported from
true_vs_s1_results.csv (searched in cwd, ./data, ../data) so the reference bands
are always present. Only S5 arms are simulated. Arms are independent ensembles
compared by a shift in the ensemble mean (Welch t-test, Cohen d); as in the
paper, only fixed-N gaps between arms are controlled.

Usage:
  # default: S5 scale sweep (full within-block scramble) -- the S4 companion
  python3 block_local_scramble_test.py

  # confound-free True->S1 titration on the whole sequence (L=N)
  python3 block_local_scramble_test.py --num-swaps 0,1,4,16,64,256,full

  # titration at a fixed locality scale
  python3 block_local_scramble_test.py --l-block 50 --num-swaps 0,2,8,32,full

  SMOKE=1 python3 block_local_scramble_test.py            # tiny sanity run
  PDTP_MAXRUN=40 python3 block_local_scramble_test.py     # cap new sims, resume
Dependencies: numpy, scipy, matplotlib
"""
import os, sys, time, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy import stats

# ------------------------------------------------------------ configuration ---
SMOKE = os.environ.get("SMOKE", "0") == "1"

def _envint(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default

ap = argparse.ArgumentParser(add_help=True)
ap.add_argument("--l-block", default=None,
                help="comma list of block lengths, or 'N' for whole sequence. "
                     "Default: scale sweep 10,50,100,500 (SMOKE: 4,16).")
ap.add_argument("--num-swaps", default=None,
                help="comma list of within-block transposition counts, tokens may "
                     "be integers or 'full'. Default: 'full'.")
ap.add_argument("--n-ens", type=int, default=None, help="ensemble size per arm")
ap.add_argument("--n", type=int, default=None, help="chain length N")
ap.add_argument("--nsteps", type=int, default=None, help="annealing steps")
ap.add_argument("--max-run", type=int, default=None, help="cap new simulations")
args = ap.parse_args()

N          = args.n      if args.n      is not None else _envint("NN",     1_000 if SMOKE else 10_000)
N_STEPS    = args.nsteps if args.nsteps is not None else _envint("NSTEPS",    60 if SMOKE else  2_000)
N_ENSEMBLE = args.n_ens  if args.n_ens  is not None else _envint("NENS",       2 if SMOKE else     40)
MAX_RUN    = args.max_run if args.max_run is not None else _envint("PDTP_MAXRUN", 10**9)
BASE_SEED  = 20260715

# force field -- identical to decomposition_test.py / finite_size_test.py / S4
B0 = 1.0; K_BOND = 45.0; K_HINGE = 0.25; ALPHA = 0.15; K_MAX = 500.0
SIGMA = 1.0; EPS_LJ = 1.0; RC_LJ = 2.0 ** (1.0 / 6.0); FCAP_LJ = 30.0
MOBILITY = 1.0; K_CONF = 0.02; DT = 0.008; MAX_DISP = 0.25
KT_HI = 0.50; KT_LO = 0.003

CSV_PATH   = "block_local_scramble_results.csv"
PRIOR_NAME = "true_vs_s1_results.csv"
REF = {"True": "#c1272d", "S1": "#0868ac"}


# ------------------------------------------------------------- job matrix -----
def _parse_swaps(spec):
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok in ("full", "f", "-1"): out.append(None)     # None => full shuffle
        elif tok != "":                out.append(int(tok))
    return out

def _parse_Ls(spec, N):
    out = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if tok in ("n", ""):           out.append(N)
        else:                          out.append(int(tok))
    return out

if args.num_swaps is None and args.l_block is None:          # default: scale sweep
    L_LIST     = [4, 16] if SMOKE else [10, 50, 100, 500]
    SWAP_LIST  = [None]                                       # full within-block shuffle
elif args.num_swaps is not None:                             # titration
    SWAP_LIST  = _parse_swaps(args.num_swaps)
    L_LIST     = _parse_Ls(args.l_block, N) if args.l_block else [N]   # L=N: confound-free
else:                                                         # only --l-block given
    L_LIST     = _parse_Ls(args.l_block, N)
    SWAP_LIST  = [None]

JOBS = [(L, s) for L in L_LIST for s in SWAP_LIST]
def _swapkey(s): return float("inf") if s is None else s
JOBS_SORTED = sorted(set(JOBS), key=lambda j: (j[0], _swapkey(j[1])))
MI = {job: 1000 + i for i, job in enumerate(JOBS_SORTED)}     # disjoint seed band


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

def mask_local_scramble(primes, N, L_block, num_swaps, rng):
    """Within-block local scramble; blocks stay in true global order.

    num_swaps is None  -> full shuffle within each block.
    num_swaps = k >= 0 -> k random within-block adjacent-pair transpositions,
                          applied independently within each contiguous block.
    Preserves gap multiset and anchor count exactly. L=1 or num_swaps=0 -> True;
    L>=len(gaps) with full shuffle -> S1.
    """
    out = np.diff(primes).copy()
    n = len(out)
    for start in range(0, n, L_block):
        end = min(start + L_block, n); m = end - start
        if m < 2: continue
        blk = out[start:end]                       # view into `out`
        if num_swaps is None:
            rng.shuffle(blk)                        # full within-block shuffle
        else:
            for _ in range(num_swaps):
                i = int(rng.integers(0, m)); j = int(rng.integers(0, m - 1))
                if j >= i: j += 1                   # distinct pair (i != j)
                tmp = blk[i]; blk[i] = blk[j]; blk[j] = tmp
    newpos = primes[0] + np.concatenate(([0], np.cumsum(out)))
    mask = np.zeros(N, dtype=bool); mask[newpos] = True
    return mask

def build_s5_arm(primes, N, L_block, num_swaps, rng):
    is_anchor = mask_local_scramble(primes, N, L_block, num_swaps, rng)
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
                if len(p) == 3: done[(p[0], int(p[1]))] = float(p[2])
    return done

def append_row(path, model, run, rho):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new: f.write("model,run,spearman_rho\n")
        f.write(f"{model},{run},{rho:.6f}\n")

def _find_prior():
    for cand in (PRIOR_NAME, os.path.join("data", PRIOR_NAME),
                 os.path.join("..", "data", PRIOR_NAME)):
        if os.path.exists(cand): return cand
    return None

def import_prior():
    src = _find_prior()
    if src is None:
        print(f"[warn] {PRIOR_NAME} not found (cwd/./data/../data) -- True/S1 "
              f"reference bands will be absent. Copy it next to this script.")
        return
    prior = load_done(src); have = load_done(CSV_PATH); added = 0
    for (m, k), rho in sorted(prior.items()):
        if m in ("True", "S1") and 0 <= k < N_ENSEMBLE and (m, k) not in have:
            append_row(CSV_PATH, m, k, rho); added += 1
    if added: print(f"imported {added} True/S1 rows from {src}")


# ---------------------------------------------------------------- tags --------
def job_tag(L, s, n_gaps):
    Lname = "N" if L >= n_gaps else str(L)
    return f"S5_L{Lname}" if s is None else f"S5_L{Lname}_s{s}"


# --------------------------------------------------------------- driver -------
def run_ensemble():
    sieve_bool = sieve_primes(N - 1); primes = np.where(sieve_bool)[0]
    n_gaps = len(primes) - 1
    import_prior(); done = load_done(CSV_PATH)
    tags = ["True", "S1"] + [job_tag(L, s, n_gaps) for (L, s) in JOBS_SORTED]
    results = {t: [None] * N_ENSEMBLE for t in tags}
    for (m, k), rho in done.items():
        if m in results and 0 <= k < N_ENSEMBLE: results[m][k] = rho
    n_have = sum(v is not None for t in tags for v in results[t])
    print(f"resume: {n_have}/{len(tags)*N_ENSEMBLE} cells (N={N}, steps={N_STEPS}, "
          f"n={N_ENSEMBLE}, n_gaps={n_gaps})")
    print(f"jobs (L, num_swaps): {[(L, ('full' if s is None else s)) for (L,s) in JOBS_SORTED]}")

    executed = 0
    for (L, s) in JOBS_SORTED:
        tag = job_tag(L, s, n_gaps); mi = MI[(L, s)]
        for k in range(N_ENSEMBLE):
            if results[tag][k] is not None: continue
            if executed >= MAX_RUN:
                print(f"(reached max-run={MAX_RUN}); rerun to continue")
                return results, tags, False
            rng = np.random.default_rng(BASE_SEED + mi * 100_000 + k)
            _, kbend = build_s5_arm(primes, N, L, s, rng)
            r = init_chain(N, rng); t0 = time.time()
            r = run_langevin(r, kbend, N_STEPS, rng); rho = core_shell_spearman(r, kbend)
            results[tag][k] = rho; append_row(CSV_PATH, tag, k, rho); executed += 1
            print(f"[{tag:>12} {k+1:02d}/{N_ENSEMBLE}] rho={rho:+.4f} ({time.time()-t0:.1f}s)")
    return results, tags, True


# --------------------------------------------------------------- analysis -----
def pair_stats(a, b):
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2) + 1e-12
    return a.mean() - b.mean(), p, (a.mean() - b.mean()) / pooled

def analyze_and_plot(results):
    arr = {t: np.array([x for x in results[t] if x is not None]) for t in results}
    have_ref = len(arr.get("True", [])) >= 2 and len(arr.get("S1", [])) >= 2
    n_gaps = len(sieve_primes(N - 1).nonzero()[0]) - 1

    print("\n=== S5 local-scramble: converse-blade sweep (Spearman rho) ===")
    if have_ref:
        T, S1 = arr["True"], arr["S1"]; span = S1.mean() - T.mean()
        print(f"  reference  True(local-order intact) = {T.mean():+.4f}+/-{T.std(ddof=1)/len(T)**.5:.4f}"
              f"   S1(fully shuffled) = {S1.mean():+.4f}+/-{S1.std(ddof=1)/len(S1)**.5:.4f}"
              f"   (span S1-True = {span:+.4f})")
        print(f"  {'arm':>14} {'rho_S5':>17} {'vs True':>18} {'vs S1':>18} {'retention':>10}")
        for (L, s) in JOBS_SORTED:
            tag = job_tag(L, s, n_gaps); v = arr.get(tag, np.array([]))
            if len(v) < 2:
                print(f"  {tag:>14}   (insufficient runs)"); continue
            gT, pT, _ = pair_stats(v, T); g1, p1, _ = pair_stats(v, S1)
            ret = (S1.mean() - v.mean()) / span if abs(span) > 1e-9 else float("nan")
            m  = f"{v.mean():+.4f}+/-{v.std(ddof=1)/len(v)**.5:.4f}"
            sT = f"{gT:+.4f} p={pT:.2f}{'*' if pT<0.05 else ' '}"
            s1 = f"{g1:+.4f} p={p1:.2f}{'*' if p1<0.05 else ' '}"
            print(f"  {tag:>14} {m:>17} {sT:>18} {s1:>18} {ret:>10.2f}")
        print("\n  retention = (rho_S1 - rho_S5)/(rho_S1 - rho_True): 1 = True-like "
              "(order intact), 0 = shuffle-like (order destroyed). Same axis as S4's 'recovery'.")
        print("  Scissors reading: S5 near retention~1 where S4 was ~0 => signal is "
              "GLOBAL/long-range, not local.")
    else:
        print("  [no True/S1 reference in CSV; printing S5 means only]")
        for (L, s) in JOBS_SORTED:
            tag = job_tag(L, s, n_gaps); v = arr.get(tag, np.array([]))
            if len(v) >= 2:
                print(f"  {tag}: rho = {v.mean():+.4f}+/-{v.std(ddof=1)/len(v)**.5:.4f} (n={len(v)})")

    # ---- adaptive figure (never crash the run) ----
    try:
        _plot(arr, have_ref, n_gaps)
    except Exception as e:
        print(f"[plot skipped: {e}]")

def _plot(arr, have_ref, n_gaps):
    Ls  = sorted(set(L for (L, s) in JOBS_SORTED))
    Ss  = sorted(set(s for (L, s) in JOBS_SORTED), key=_swapkey)
    scale_sweep = (len(Ss) == 1) and (len(Ls) > 1)
    titration   = (len(Ls) == 1) and (len(Ss) > 1)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    if have_ref:
        for name in ("True", "S1"):
            a = arr[name]; m = a.mean(); se = a.std(ddof=1) / np.sqrt(len(a))
            ax.axhline(m, color=REF[name], ls="--", lw=1.7,
                       label=f"{name} " + ("(local order intact)" if name == "True" else "(fully shuffled)"))
            ax.fill_between([1e-9, 1e9], m - se, m + se, color=REF[name], alpha=0.13, zorder=0)

    def arm_ms(pred):
        xs, ys, es = [], [], []
        for (L, s) in JOBS_SORTED:
            if not pred(L, s): continue
            v = arr.get(job_tag(L, s, n_gaps), np.array([]))
            if len(v) >= 2:
                xs.append((L, s)); ys.append(v.mean()); es.append(v.std(ddof=1) / np.sqrt(len(v)))
        return xs, np.array(ys), np.array(es)

    if scale_sweep:
        xs, ys, es = arm_ms(lambda L, s: True)
        X = np.array([L for (L, s) in xs], float)
        ax.errorbar(X, ys, yerr=es, marker="o", ms=9, lw=2.2, capsize=4,
                    color="#2ca25f", zorder=5, label="S5 local scramble (full)")
        ax.set_xscale("log"); ax.set_xlim(X.min() * 0.6, X.max() * 1.8)
        ax.set_xlabel(r"block length $L_{\mathrm{block}}$ (gaps)")
        ax.set_title("S5 converse blade: does local scrambling destroy the signal?")
    elif titration:
        xs, ys, es = arm_ms(lambda L, s: True)
        finite = [(t[1], y, e) for t, y, e in zip(xs, ys, es) if t[1] is not None]
        hasfull = any(t[1] is None for t in xs)
        if finite:
            fx = np.array([max(k, 0.5) for k, _, _ in finite])  # 0 -> 0.5 for log axis
            fy = np.array([y for _, y, _ in finite]); fe = np.array([e for _, _, e in finite])
            ax.errorbar(fx, fy, yerr=fe, marker="o", ms=8, lw=2.2, capsize=4,
                        color="#2ca25f", zorder=5, label="S5 (k within-block swaps)")
            xr = fx.max() * 3
        else:
            xr = 10.0
        if hasfull:
            t = next(t for t in xs if t[1] is None); idx = xs.index(t)
            ax.errorbar([xr], [ys[idx]], yerr=[es[idx]], marker="s", ms=9, capsize=4,
                        color="#006d2c", zorder=6, label="S5 full shuffle")
        ax.set_xscale("log")
        ax.set_xlabel(r"within-block transpositions $k$ (per block; 0$\to$True, 'full'$\to$S1)")
        L0 = Ls[0]; ax.set_title(f"S5 titration at L={'N' if L0>=n_gaps else L0}: walking True $\\to$ S1")
    else:
        # generic: index arms on x
        xs, ys, es = arm_ms(lambda L, s: True)
        X = np.arange(len(xs))
        ax.errorbar(X, ys, yerr=es, marker="o", ms=8, lw=2, capsize=4, color="#2ca25f",
                    zorder=5, label="S5 arms")
        ax.set_xticks(X); ax.set_xticklabels([job_tag(L, s, n_gaps) for (L, s) in xs],
                                             rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"Spearman $\rho$ (stiffness vs radius)")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout(); plt.savefig("block_local_scramble_plot.pdf"); plt.close(fig)
    print("wrote block_local_scramble_plot.pdf")


if __name__ == "__main__":
    t0 = time.time()
    results, tags, complete = run_ensemble()
    if complete:
        analyze_and_plot(results); print(f"done in {time.time()-t0:.1f}s -> {CSV_PATH}")
    else:
        print(f"[partial] saved {CSV_PATH}; rerun to continue. {time.time()-t0:.1f}s")
