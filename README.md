# Prime-Driven Topological Polymer (PDTP)

**Beyond the marginal distribution: long-range correlations in prime gaps imprint on the 3D conformation of a self-avoiding polymer**

Ruqing Chen — GUT Geoservice Inc., Montreal, Quebec, Canada — <ruqing@hotmail.com>

This repository contains the code, data, and figures reproducing the paper.

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
│   └── make_figures.py       regenerate all figures (PDF) from data/
├── data/
│   ├── true_vs_s1_results.csv        base True/S1 ensemble, N=10000, n=40 (Spearman ρ)
│   ├── decomposition_results.csv     True/S1/S3, N=10000, n=40 (Spearman ρ)
│   ├── finite_size_results.csv       True/S1, N in {10k,20k,40k}, n=20 (ρ, ΔR)
│   ├── decisive_40k_results.csv      True/S1, N=40000 deep anneal, n=40 (ρ, ΔR)
│   ├── conformation_coords.npy       final coordinates (N,3) of Fig. 1
│   └── conformation_kbend.npy        per-bead stiffness k_i of Fig. 1
└── figures/
    ├── fig1_conformation.pdf
    ├── fig2_decisive.pdf
    ├── fig3_decomposition.pdf
    └── fig4_finite_size.pdf
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

If you use this code or data, please cite the paper (preprint) and this repository.
A `CITATION.cff` / archived DOI (e.g. via Zenodo) will be added upon release.

## License

MIT — see `LICENSE`.
