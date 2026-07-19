# -*- coding: utf-8 -*-
"""
Pre-flight numerical-stability calibration (paper Section III).

The gap-driven stiffness of Eq. (1) is amplified exponentially, so the rare
long-gap sites carry very large bending moduli. We bound the parameters a priori
with a worst-case stress test on the isolated segment containing the global
maximum prime gap, run WITHOUT the production displacement cap so that any latent
divergence is exposed. Three findings are reproduced:

  (A) alpha sweep in [0.05, 0.15]: stable (no divergence).
  (B) k_max sweep up to the fully uncapped value (k ~ 1.2e4 at g_max): still
      stable -- the cosine bending force is self-limiting (F ~ sin(theta) -> 0
      as a rod straightens), so the amplification is NOT the binding constraint.
  (C) dt sweep at fixed forces: stable for dt <= 0.01, diverges for dt >~ 0.02,
      set by the stiffest harmonic (bond) mode, dt < 2/(4 mu k_bond) ~ 0.011.
      Production dt = 0.008 sits a factor ~2.5 below threshold.

Usage:  python3 calibration.py
Dependencies: numpy
"""
import numpy as np

RNG = np.random.default_rng(0)

N        = 40_000     # production index range (fixes g_max)
K_HINGE  = 0.25
K_BOND   = 45.0
B0       = 1.0
ALPHA    = 0.15
MOBILITY = 1.0
KT       = 0.05       # small thermal noise, continually injects curvature
N_STEPS  = 2000
SEG_LEN  = 150
BLOWUP   = 1e4        # |r| beyond this -> divergence


def sieve_primes(n):
    ip = np.ones(n + 1, dtype=bool); ip[:2] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if ip[i]: ip[i * i::i] = False
    return ip


def prime_gap_array(is_prime, N):
    primes = np.where(is_prime)[0]
    g = np.zeros(N, dtype=float)
    for a, b in zip(primes[:-1], primes[1:]):
        if b - a > 1: g[a + 1:b] = b - a
    g[0:primes[0]] = primes[0]
    g[primes[-1] + 1:N] = N - primes[-1]
    return g, primes


def seg_forces(r, kseg):
    """Isolated segment: bond + gap-driven cosine bending only."""
    n = r.shape[0]; F = np.zeros_like(r)
    d = r[1:] - r[:-1]
    L = np.linalg.norm(d, axis=1) + 1e-12
    fb = (K_BOND * (L - B0) / L)[:, None] * d
    F[0:n - 1] += fb; F[1:n] -= fb
    a = r[1:-1] - r[:-2]; b = r[2:] - r[1:-1]
    na = np.linalg.norm(a, axis=1) + 1e-12; nb = np.linalg.norm(b, axis=1) + 1e-12
    ah, bh = a / na[:, None], b / nb[:, None]
    cos = np.clip(np.sum(ah * bh, axis=1), -1.0, 1.0)
    kk = kseg[1:-1][:, None]
    dda = (bh - cos[:, None] * ah) / na[:, None]
    ddb = (ah - cos[:, None] * bh) / nb[:, None]
    F[0:n - 2] += -kk * dda; F[2:n] += kk * ddb; F[1:n - 1] += kk * (dda - ddb)
    return F


def init_segment(L):
    r = np.zeros((L, 3)); r[:, 0] = np.arange(L) * B0
    r[:, 1] = 0.5 * np.sin(np.pi * np.arange(L) / L)   # gentle initial bend
    r += 0.02 * RNG.standard_normal(r.shape)
    return r


def stress_test(kseg, dt, n_steps=N_STEPS):
    """Run isolated segment WITHOUT displacement cap; return (crashed, step)."""
    r = init_segment(len(kseg))
    for step in range(n_steps):
        F = seg_forces(r, kseg)
        if not np.all(np.isfinite(F)):
            return True, step
        r += MOBILITY * F * dt + np.sqrt(2 * MOBILITY * KT * dt) * RNG.standard_normal(r.shape)
        if (not np.all(np.isfinite(r))) or np.abs(r).max() > BLOWUP:
            return True, step
    return False, None


def main():
    is_prime = sieve_primes(N - 1)
    g, primes = prime_gap_array(is_prime, N)
    imax = int(np.argmax(g)); gmax = int(g[imax])
    p_left = primes[np.searchsorted(primes, imax, "right") - 1]
    p_right = primes[np.searchsorted(primes, imax, "left")]
    center = (p_left + p_right) // 2
    s0 = max(0, center - SEG_LEN // 2); s1 = min(N, s0 + SEG_LEN); s0 = max(0, s1 - SEG_LEN)
    gseg = g[s0:s1]
    print(f"g_max = {gmax}  between primes {p_left}-{p_right};  segment [{s0},{s1})")

    print("\n(A) alpha sweep  (k_max=1000, dt=0.008):")
    for alpha in np.round(np.arange(0.05, 0.1501, 0.01), 2):
        kseg = np.minimum(K_HINGE * np.exp(alpha * gseg), 1000.0); kseg[gseg == 0] = K_HINGE
        cr, st = stress_test(kseg, 0.008)
        print(f"   alpha={alpha:.2f}  max k={kseg.max():8.1f}  {'CRASH@'+str(st) if cr else 'stable'}")

    print("\n(B) k_max sweep  (alpha=0.15, dt=0.008; uncapped ~ 1.2e4 at g_max):")
    for kmax in [500, 1000, 4000, 12000, 32000]:
        kseg = np.minimum(K_HINGE * np.exp(ALPHA * gseg), kmax); kseg[gseg == 0] = K_HINGE
        cr, st = stress_test(kseg, 0.008)
        print(f"   k_max={kmax:6d}  max k={kseg.max():9.1f}  {'CRASH@'+str(st) if cr else 'stable'}")

    print("\n(C) dt sweep  (alpha=0.15, k_max=500):")
    kseg = np.minimum(K_HINGE * np.exp(ALPHA * gseg), 500.0); kseg[gseg == 0] = K_HINGE
    crash_dt = None
    for dt in [0.008, 0.01, 0.02, 0.05, 0.10, 0.20]:
        cr, st = stress_test(kseg, dt, n_steps=3000)
        if cr and crash_dt is None: crash_dt = dt
        print(f"   dt={dt:.3f}  {'CRASH@'+str(st) if cr else 'stable'}")
    if crash_dt:
        print(f"\n   first divergence at dt ~ {crash_dt};  production dt=0.008 margin ~ {crash_dt/0.008:.1f}x")


if __name__ == "__main__":
    main()
