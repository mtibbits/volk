# Kernel-correctness harness — full-catalog triage snapshot (2026-06-14)

The **seeded, severity-ranked** regeneration of the epic-#85 snapshot, taken after
the harness was completed by the WP 2.1 gate: per-kernel QA params in the
impl-mode sweep (#106 Phase A / PR #140), `HARNESS_SEED` reproducibility (#134),
the per-impl `max_err` severity column (#135), and ref-mode fail-closed→skip
(#133). It supersedes the unseeded `2026-06-11` snapshot and is the corrected
per-kernel fix backlog for the #105 campaign.

## Provenance

- Branch: `chore/106-snapshot-regen-v2` off `dev/all-prs` @ `e7e90cc`
- Host: 11th Gen Intel Core i7-1165G7 (AVX-512-capable, 8 threads) — impl
  coverage is CPU-dependent; this host exercises the widest impl set.
- Build: Release, `-DENABLE_TESTING=ON`, GCC, Linux (`build-allprs`).
- Command:
  `HARNESS_SEED=20260614 python3 tools/run_harness_triage.py --build-dir build-allprs --out docs/kernel_correctness_harness/triage-2026-06-14.csv --seed 20260614 --modes sweep,canary,immutable,misaligned`
- **Seed: 20260614** (pinned, reproducible — #134). The CSV is one seeded
  sample; the seed-stability of every finding is characterised by the 10-seed
  study below.
- Format: **v2** (`# volk-harness-report v2` marker + 6-column
  `kernel,impl,mode,result,failed_vlens,max_err`).

## Result distribution (2,854 rows = kernel × impl × mode)

| result  | rows | meaning |
|---------|------|---------|
| ok      | 2487 | impl passed that mode at every vlen |
| partial | 167  | canary: in-bounds output unwritten — the documented reduction/index-kernel pattern, not a defect |
| skip    | 154  | puppet / no guardable buffer / no qa entry (and ref-mode unsupported-shape, #133) |
| FAIL    | 46   | findings: 8 `ref` (oracle divergence) + 35 `impl` (sweep divergence) + 3 `misaligned` |

**The headline:** 46 finding rows, down from the `2026-06-11` snapshot's **113**
(112 FAIL + 1 abort). Phase A (impl-mode sweep now consumes each kernel's
registered tol/scalar/absolute_mode) cleared ~67 rows of harness artifact — the
old snapshot was scoring impls against the driver's 1e-6 default instead of the
kernel's own adjudicated contract (e.g. reciprocal's `make_tol(6.15e-5)`, the
reductions' `make_absolute(…)`, clamppuppet's real scalar vs the degenerate
`327`).

## Severity ranking (the new `max_err` column, #135)

Worst-case per-impl divergence, now machine-readable. The ref-mode findings:

| max_err | kernel / impl | issue |
|---|---|---|
| 6.45e+05 | `volk_32f_tanh_32f` / `series` | #108 (stale-pointer; catastrophic) |
| 3.8 | `volk_32f_tanh_32f` / `a_avx`,`u_avx`,`*_fma` | #108 (tail/edge) |
| 2 | `volk_32f_tanh_32f` / `a_sse`,`u_sse` | #108 |
| 1.85 | `volk_32fc_s32f_power_32fc` / `generic` | #107 (swapped-atan2; every vlen ≥2) |

`volk_32f_invsqrt_32f` (impl mode) reaches `inf` (#112, special-value). The
column lets the campaign rank by magnitude instead of treating a `6.45e+05` and
a `3e-05` divergence as the same bare `FAIL`.

## Seed-stability study (10 seeds, sweep mode)

Because near-tolerance findings vary with the random data, each sweep finding
was re-run across **10 seeds** (`20260614, 1, 7, 137, 99991, 2718281, 42,
1000003, 31337, 8675309`). This is the load-bearing classification — **triage by
this table, not by the single-seed CSV** (a margin-flicker can fire in any one
sample, as `conjugate_dot_prod` does at seed 20260614).

### Stable genuine findings — FAIL at 10/10 seeds (keep open)
`volk_32f_tanh_32f` (#108), `volk_32fc_s32f_power_32fc` (#107),
`volk_32f_atan_32f` (#109), `volk_32f_tan_32f` (#110),
`volk_32f_invsqrt_32f` (#112), `volk_32f_8u_polarbutterflypuppet_32f` (#116),
`volk_32f_stddev_and_mean_32f_x2` (#125), `volk_32fc_index_max_16u` (#127),
`volk_32fc_index_max_32u` (#128).

### Reduction-family margin flickers — FAIL at 1–3/10 seeds (→ #118 contract)
`volk_32fc_x2_conjugate_dot_prod_32fc` (3/10, #120),
`volk_32fc_accumulator_s32fc` (3/10, #124),
`volk_16ic_x2_dot_prod_16ic` (3/10, no child yet),
`volk_32fc_32f_dot_prod_32fc` (2/10, #121),
`volk_32f_x2_dot_prod_32f` (1/10, #118),
`volk_32fc_x2_dot_prod_32fc` (1/10, #119).
These cross their registered `make_absolute(…)` tolerance only on some random
draws — the accumulation-order question #118 exists to decide. They are neither
stable defects nor clean artifacts; their adjudication belongs to the
reduction-tolerance contract (#118), not to closing them here.

### Stable-clear under the fixed harness — FAIL at 0/10 seeds (harness artifacts)
`volk_32f_expfast_32f` (#111), `volk_32f_reciprocal_32f` (#113),
`volk_32f_s32f_clamppuppet_32f` (#114), `volk_32f_x2_powpuppet_32f` (#115),
`volk_32fc_s32f_magnitude_16i` (#117) — each had a registered per-kernel
tol/scalar the old driver ignored; with Phase A applying it they pass at every
seed. Also 0/10: the loose-tolerance reductions `volk_16i_32fc_dot_prod_32fc`
(#122), `volk_32f_accumulator_s32f` (#123), `volk_32f_s32f_calc_spectral_noise_floor_32f`
(#126) — robustly within their `make_absolute(1e-1 / 2e-2)` contracts.

## Other-mode findings (outside the sweep re-triage)

- **Misaligned (#91): `volk_32f_x3_sum_of_poly_32f`** FAILs (generic + u_avx +
  u_avx_fma) at tiny vlens — this is the **#98 parameter-array under-seed**, not
  a Phase-A param artifact; tracked separately. All other kernels: 0 misaligned
  findings (puppets excluded by design — the rotator2 `u_avx512f` fault, #101, is
  reachable only via its puppet, structurally outside this mode).
- **Input immutability (#90): 0 findings.**
- **Canary (#89): 0 guard violations**; 167 `partial` rows are the documented
  fixed-size-output reduction/index pattern (note #98 when triaging sum_of_poly).

## Re-triage outcome for the #105 campaign

| Bucket | Count | Children | Action |
|---|---|---|---|
| Stable genuine (10/10) | 9 | #107 #108 #109 #110 #112 #116 #125 #127 #128 | keep open (per-kernel fix tickets) |
| Reduction margin (1–3/10) | 6 kernels | #118 #119 #120 #121 #124 (+ 16ic dot_prod, unfiled) | resolve under #118's tolerance contract |
| Harness artifact (0/10) | 8 | #111 #113 #114 #115 #117 (clear-cut) + #122 #123 #126 (reduction, loose-tol) | close as harness-artifact (reductions: prefer closing with #118) |

The conv_k7 SIGABRT from the old snapshot is gone (fixed on `dev/all-prs` by
PR #100; the old baseline predated it).

## Reproducing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=ON
cmake --build build -j"$(nproc)"
HARNESS_SEED=20260614 python3 tools/run_harness_triage.py --build-dir build \
  --out triage.csv --seed 20260614 --modes sweep,canary,immutable,misaligned
```

Retention: latest snapshot only — this replaces `triage-2026-06-11.*` (history
in git).
