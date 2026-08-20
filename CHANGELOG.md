# Changelog

## Version 3.0 (2026-08-19) — consistency, protocol-dependence, and reporting revision

**No v2.0 datum is altered and no new production ensemble is introduced.** All
changes are to reporting, statistical bookkeeping, and archival. Summary of the
paper-level changes (see also the *Note (Version 3.0)* on p. 2 of the paper):

### Statistics and reporting
- **Error-bar convention stated explicitly.** All `x ± y` values are ensemble
  mean ± s.e.m.; box plots are medians and quartiles; p-values are two-sided
  Welch t-tests unless Mann–Whitney is stated; d is Cohen's d (Sec. 4 preamble).
- **Annealing depth recorded per ensemble.** Every figure caption now states the
  annealing depth in absolute steps *and* steps per bead:
  - Fig. 2 (decisive N = 4×10⁴): 8000 steps = 0.20 steps/bead.
  - Fig. 3 (decomposition, N = 10⁴): 2000 steps = 0.20 steps/bead (reference
    protocol, depth-matched to Fig. 2).
  - Fig. 4 (finite-size sweep): 2000/3500/5000 steps = 0.20/0.175/0.125
    steps/bead — **not** depth-matched across sizes; its N = 10⁴ point uses an
    n = 20 ensemble distinct from Fig. 3's n = 40 ensemble, which explains the
    small offset between the two figures.
  - Fig. 5 (S4/S5 scissors): 2000 steps = 0.20 steps/bead.
- **R_g comparison power flagged.** The R_g true-vs-shuffle comparison uses the
  initial n = 10 sweep and only bounds, rather than excludes, an R_g effect
  (Sec. 4.1); the invariance claim is correspondingly softened to "within our
  sensitivity."

### Correlation length ξ
- **ξ identified as a protocol-conditioned observable.** Eq. (8) is the
  half-retention scale under the reference protocol (2000 steps, 0.20
  steps/bead), not a protocol-independent equilibrium constant (Sec. 5.2,
  abstract, Conclusion).
- **Note added (Version 3.0) in Sec. 5.2.** Follow-up work by the author
  (companion manuscript, in preparation), which measures ξ(N) across
  N = 10⁴–8×10⁴ at annealing depth matched across sizes and deeper than the
  reference protocol, re-measures the N = 10⁴ crossover under that protocol and
  obtains ξ ≈ 4.8×10² gaps (68% bootstrap CI [4.1, 5.7]×10²; ξ/n_gap ≈ 0.39),
  at the upper edge of Eq. (8)'s interval. Both values are macroscopic
  (~0.3–0.4 of the sequence); the falsification of the local-trap hypothesis
  and the macroscopic-order conclusion are unaffected. The Conclusion is
  updated to state that the macroscopic *character* of ξ, not its precise
  value, is the protocol-robust statement.

### Model definition
- **Length–stiffness coupling stated.** Because every bead of an inter-prime
  run inherits the same gap value, a long gap yields a segment that is both
  longer (contour ∝ g) and stiffer (modulus ∝ e^{αg}); this deliberate
  compounding is now stated in Sec. 2.1, together with the observation that
  every surrogate inherits the identical coupling, so all true-vs-surrogate
  contrasts compare like with like.
- **Boundary treatment quantified.** The chain-end conventions (g_i = p₁ for
  i < p₁; g_i = N − p_last for i > p_last) affect a fraction ≲ 10⁻³ of beads
  and are immaterial to the rank statistics (Sec. 2.1).

### Limitations (two additions)
- ξ protocol dependence added as an explicit limitation.
- Two inexpensive robustness controls named as next steps: (i) partial rank
  correlation controlling for backbone index i (plus a sequence-reversal
  control), addressing the monotone 1/ln i stiffness drift along the chain;
  (ii) a spectrally matched (IAAFT-type) surrogate to test whether the fold
  reads only second-order statistics.

### References
- Added M. Wolf, Physica A 241, 493 (1997) (1/f-type long-range correlations
  in the primes), cited in Sec. 5.3.
- Added the companion manuscript (in preparation, 2026) supporting the
  Sec. 5.2 note added.

### Related-work positioning (novelty audit)
A literature check identified five bodies of prior art that v2.0 did not engage.
The Introduction and Sec. 5.3 ("Connections and relation to prior work") now cite
and position the paper against each:
- **Torquato, Zhang & de Courcy-Ireland** (J. Stat. Mech. 2018, 093401; J. Phys. A
  52, 135002, 2019): the primes are effectively limit-periodic and hyperuniform —
  independent evidence of macroscopic order in the primes. Sec. 5.3 adds the
  conjecture that the fold registers a residue of this multiscale order and notes
  that the IAAFT spectral surrogate (Limitations) tests it directly, since
  hyperuniformity is a two-point property.
- **Khokhlov & Khalatur** (PRL 82, 3456, 1999) and **Govorun et al.** (PRE 64,
  040903(R), 2001): conformation-dependent sequence design — the converse
  (structure→sequence) map, whose designed sequences carry Lévy-flight LRC coupled
  to core–shell organization. Sec. 5.3 contrasts adapted vs. quenched correlations
  and notes the sign of the LRC effect on radial sorting depends on correlation
  structure.
- **Mamasakhlisov, Morozov & Shahinian** (cond-mat/9802225, 1998): power-law
  correlated random heteropolymers reshape folding — ensemble-level prior;
  differentiated by the single-sequence, marginal-matched surrogate setting here.
- **Azevedo et al.** (Physica A 445, 27, 2016): Fibonacci-arranged DNA chains —
  prior deterministic-aperiodic sequence→molecular chain mapping; differentiated
  by the ordering-vs-composition attribution, which was not posed there.
- **Odijk** (arXiv:1904.09693, 2019): polymeric compaction proposed as a probe of
  long-range correlations in data sequences — anticipates the "polymer as probe"
  framing; differentiated by the mechanical (bending-field) map, rank-statistics
  readout, and the scale-selective surrogate hierarchy.
The "largely unexplored regime" claim in the Introduction is correspondingly
tempered to "controlled attribution has been lacking."

### Typesetting and archival
- **PDF text layer fixed.** Added `cmap` + `glyphtounicode` (+ `lmodern` when
  available): fi/ff/ffi ligatures now copy, search, and extract correctly
  (previously "stiffness" extracted as "stiness", "shuffle" as "shue").
- **Zenodo archival.** Data-availability section now records the Zenodo
  concept DOI and the version-3.0 record DOI (placeholders
  `10.5281/zenodo.XXXXXXX` / `10.5281/zenodo.YYYYYYY` — fill in after minting
  the release). `CITATION.cff` bumped to 3.0.0.

### Action items for the author before publishing v3 on Zenodo
1. ~~Mint the Zenodo release and fill in the DOI~~ — DONE: v3.0 is archived
   under DOI 10.5281/zenodo.22020688 (filled into Data availability and
   CITATION.cff).
2. The companion manuscript is cited inline in the Sec. 5.2 note added
   ("R. Chen, companion manuscript, in preparation") rather than as a numbered
   reference, since unpublished work should not appear in the reference list.
   Once the second paper is posted, promote it to a numbered reference with its
   arXiv ID / DOI.
3. Optional but recommended: run the two robustness controls named in
   Limitations (partial ρ | index; IAAFT surrogate) — both reuse the existing
   pipeline at N = 10⁴, n = 40, ≈ same cost as one scissors arm.

## Version 2.0 (2026-07-19) — mechanism-falsification update
- Local "topological-trap" hypothesis of v1 tested and falsified by the S4
  block shuffle and S5 local scramble; ordering localized to a macroscopic
  correlation length ξ ≈ 3.5×10² gaps at N = 10⁴ (Sec. 5.2, Fig. 5).
- All v1.0 data and results unchanged.

## Version 1.0 (2026-07-19)
- Initial release.
