# Prime-Driven Topological Polymer (PDTP)

**Beyond the marginal distribution: long-range correlations in prime gaps imprint on the 3D conformation of a self-avoiding polymer**

Ruqing Chen — GUT Geoservice Inc., Montreal, Quebec, Canada — <ruqing@hotmail.com>

This repository contains the code, data, and figures reproducing the paper.

> **Version 2.0 (mechanism-falsification update).** The local “topological-trap”
> mechanism proposed in v1.0 has been **tested and falsified** by two new
> scale-selective surrogate experiments (S4 block shuffle, S5 local scramble). They
> localize the ordering to a **macroscopic correlation length ξ ≈ 355 gaps**
> (≈30% of the sequence). See *Version 2.0* below and paper §5.2. All v1.0 data
> and results are unchanged.

---

## Summary

We map the arithmetic structure of the primes onto a coarse-grained polymer: the
local bending stiffness of bead *i* is an exponential function of its local prime
gap *g_i*,

```
k_i = k_hinge                              if i is prime
k_i = min( k_hinge * exp(alpha * g_i), k_max )   if i is composite
```

so dense prime clusters become flexible hinges and long inter-prime runs become
rigid rods. Under overdamped Langevin dynamics with simulated annealing, we ask
whether the folded state depends on the **composition** of the stiffness sequence or
on its **arithmetic ordering**. To separate the two we use surrogate null models that
preserve the marginal stiffness distribution exactly while destroying sequential
order.

**Main findings.** Gross morphology (radius of gyration `R_g`, tail-weighted
core–shell separation `ΔR`) is surrogate-invariant. The Spearman rank correlation
`ρ` between stiffness and radial position is significantly *suppressed* for the true
prime sequence relative to a shuffle (`ρ = +0.105` vs. `+0.137` at `N = 40000`;
Welch `p = 6.5e-5`, Cohen `d = -0.96`). A three-way decomposition attributes this to
ordering rather than to the gap distribution. The effect is quantitatively modest
(~1% of the radial-rank variance) but statistically robust and reproducible across
system sizes.

**Version 2.0 — what drives the ordering (scissors test).** A complementary pair of
scale-selective surrogates, each preserving the gap multiset exactly, cut the
correlation length from opposite sides. **S4** (block shuffle) keeps local gap order
within blocks of length `L` but shuffles the blocks globally; **S5** (local scramble)
keeps the blocks in true global order but scrambles gaps within each block.
Destroying **local** order (S5) leaves the signal fully intact — indistinguishable
from the true chain up to `L ≈ 100` gaps (`p = 0.62`–`0.91`) — while destroying
**global** order (S4) abolishes it at every scale (`p ≥ 0.10` vs. shuffle). Local
order is thus neither necessary nor sufficient: the **local twin-prime trap
hypothesis is ruled out**. The starvation-free S5 blade measures a **macroscopic
correlation length `ξ ≈ 355` gaps** (68% CI `[180, 490]`), spanning ~30% of the
sequence — the true driver of the topological imprint.

---

## Repository layout

```
.
├── paper/
│   ├── pdtp.tex              full manuscript (LaTeX, article class)
│   └── pdtp.pdf              compiled PDF
├── code/
│   ├── calibration.py        pre-flight numerical-stability calibration (Sec. III)
│   ├── conformation.py       3D conformation, reduced force field (Fig. 1)
│   ├── decisive_40k_test.py  decisive N=40000 deep-anneal test, True vs S1 (Fig. 2)
│   ├── decomposition_test.py True/S1/S3 ordering-vs-marginal decomposition (Fig. 3)
│   ├── finite_size_test.py   finite-size sweep N=10k/20k/40k, True vs S1 (Fig. 4)
│   ├── block_shuffle_test.py       S4 block shuffle: destroy global, keep local (Fig. 5)
│   ├── block_local_scramble_test.py S5 local scramble: destroy local, keep global (Fig. 5)
│   ├── make_scissors_figure.py     regenerate Fig. 5 from the S4/S5/baseline CSVs
│   └── make_figures.py       regenerate figures 1-4 (PDF) from data/
├── data/
│   ├── true_vs_s1_results.csv        base True/S1 ensemble, N=10000, n=40 (Spearman ρ)
│   ├── decomposition_results.csv     True/S1/S3, N=10000, n=40 (Spearman ρ)
│   ├── finite_size_results.csv       True/S1, N in {10k,20k,40k}, n=20 (ρ, ΔR)
│   ├── decisive_40k_results.csv      True/S1, N=40000 deep anneal, n=40 (ρ, ΔR)
│   ├── block_shuffle_results.csv        S4 arms, N=10000, n=40 (Spearman ρ)
│   ├── block_local_scramble_results.csv S5 arms, N=10000, n=40 (Spearman ρ)
│   ├── conformation_coords.npy       final coordinates (N,3) of Fig. 1
│   └── conformation_kbend.npy        per-bead stiffness k_i of Fig. 1
└── figures/
    ├── fig1_conformation.pdf
    ├── fig2_decisive.pdf
    ├── fig3_decomposition.pdf
    ├── fig4_finite_size.pdf
    └── fig5_scissors.pdf
```

## Reproducing the figures

With the provided data files:

```bash
cd code
python3 make_figures.py      # writes figures/fig1..fig4.pdf
```

## Reproducing the data (from scratch)

Each experiment writes a CSV incrementally and resumes automatically if interrupted.
Large runs can be chunked with `PDTP_MAXRUN`, and every script accepts `SMOKE=1` for a
tiny sanity-check run.

```bash
cd code
python3 calibration.py                          # Sec. III stability numbers (seconds)
python3 conformation.py                         # Fig. 1 (~2.5-3 min, N=40000)
python3 decomposition_test.py                   # Fig. 3 (True/S1/S3, N=10000, n=40)
PDTP_MAXRUN=20 python3 finite_size_test.py      # Fig. 4 (rerun until complete)
PDTP_MAXRUN=10 python3 decisive_40k_test.py     # Fig. 2 (rerun until complete; ~7-8 h total)

# Version 2.0 scissors surrogates (Fig. 5). Both auto-import true_vs_s1_results.csv
# (cwd / ./data / ../data) for the reference bands, and resume if interrupted.
python3 block_shuffle_test.py                   # S4 scale sweep L=10,50,100,500 (~2 h, n=40)
python3 block_local_scramble_test.py            # S5 scale sweep (converse blade)
python3 block_local_scramble_test.py --num-swaps 0,1,4,16,64,256,full  # confound-free True->S1 titration
python3 make_scissors_figure.py                 # writes figures/fig5_scissors.pdf
```

Notes:
- `decomposition_test.py` will import an existing `true_vs_s1_results.csv` (same
  seeding) and compute only the S3 arm, if that file is present in the working dir.
- Absolute `ρ`/`ΔR` depend on the annealing protocol; only the fixed-`N`
  True-vs-shuffle *gap* is a controlled comparison (see paper Sec. IV D).

## Requirements

Python 3.9+, with `numpy`, `scipy`, `matplotlib`. See `requirements.txt`.

```bash
pip install -r requirements.txt
# or:  pip install numpy scipy matplotlib
```

## Building the paper

```bash
cd paper
pdflatex pdtp.tex && pdflatex pdtp.tex
```

## Citation

If you use this code or data, please cite the paper and this repository (see
`CITATION.cff`, version 2.0.0). The archived Zenodo record is updated to Version 2.0
under the original concept DOI.

## License

MIT — see `LICENSE`.
