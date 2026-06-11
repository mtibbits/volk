# Kernel-correctness harness — full-catalog triage snapshot (2026-06-11)

One-shot snapshot of the assembled epic-#85 harness across the full kernel
catalog. This is the per-kernel fix backlog: every FAIL/abort row below becomes
its own fix ticket, sliced by kernel (the epic's out-of-scope boundary — no
fixes ship with the harness itself).

## Provenance

- Branch: `chore/92-harness-closeout-negative-control` @ `5936601` (+ the #92
  harness changes, working tree)
- Host: 11th Gen Intel Core i7-1165G7 (AVX-512-capable, 8 threads) — kernel ×
  impl coverage is CPU-dependent; an AVX-512 host exercises the widest impl set
- Build: Release, `-DENABLE_TESTING=ON`, GCC, Linux
- Command: `python3 tools/run_harness_triage.py --build-dir build
  --out docs/kernel_correctness_harness/triage-2026-06-11.csv`
- Wall time: 8.9 s (616 runs, 8 workers; one process per kernel × mode)
- **Reproducibility:** qa test data is re-randomized every run
  (`load_random_data` seeds from `std::random_device`), so this snapshot is
  one sample. Gross findings (the two motivating defects, the conv_k7 abort,
  the structural canary partials) are stable across runs; near-tolerance
  findings — the reduction/accumulator family's marginal divergences — vary
  in their exact failing vlens and can appear/disappear at the margin (e.g.
  `volk_32fc_x2_dot_prod_32fc` sse3 and the sum_of_poly misaligned rows
  flipped between verification re-runs). Triage by kernel set, not exact
  vlen lists.

## Result distribution (2,796 rows = kernel × impl × mode)

| result  | rows | meaning |
|---------|------|---------|
| ok      | 2364 | impl passed that mode at every vlen |
| partial | 165  | canary: in-bounds output elements unwritten — expected for reduction/index kernels (fixed-size scalar output); surfaced for triage, not a defect class by itself |
| skip    | 154  | puppet (canary/immutable/misaligned modes), no guardable buffer, or no qa entry |
| FAIL    | 112  | a real finding: 8 `ref` rows (divergence from the independent double-precision oracle) + 104 `impl` rows (impl-vs-generic divergence in the tail/edge sweep) |
| abort   | 1    | the run died and was recorded as a finding (process isolation kept the snapshot alive) |

## Headline findings (the motivating escaped defects, still live on this branch)

- `volk_32fc_s32f_power_32fc` — `ref` FAIL at every vlen ≥ 2: the swapped-`atan2f`
  + negated-cosine defect (evidence commit `e29bc7f`). Its single generic impl
  means impl-vs-impl scores it `ok` — exactly the #88 blind spot.
- `volk_32f_tanh_32f` — `ref` FAIL: the `series` impl fails at every vlen ≥ 7
  (stale input pointer in the clamp branches, `e29bc7f`); the SSE/AVX impls fail
  only at the tail-remainder vlens (7, 10–15) where their scalar-tail code hits
  the clamp edges. Generic (`tanhf`) passes against the oracle.

## Abort findings (tiny-vlen / hard-abort class)

- `volk_8u_conv_k7_r2puppet_8u` — sweep run dies with SIGABRT (static decision
  buffer overflow at small vlens). Already fixed on `dev/all-prs` by PR #100;
  this stack's baseline predates that fix, so it appears here as a live finding.

## FAIL kernels (22 in this sample; full per-impl × per-vlen detail in the CSV)

`volk_16i_32fc_dot_prod_32fc`, `volk_32f_8u_polarbutterflypuppet_32f`,
`volk_32f_accumulator_s32f`, `volk_32f_atan_32f`, `volk_32fc_32f_dot_prod_32fc`,
`volk_32fc_accumulator_s32fc`, `volk_32fc_index_max_16u`,
`volk_32fc_index_max_32u`, `volk_32fc_s32f_magnitude_16i`,
`volk_32fc_s32f_power_32fc`, `volk_32fc_x2_conjugate_dot_prod_32fc`,
`volk_32fc_x2_dot_prod_32fc`, `volk_32f_expfast_32f`, `volk_32f_invsqrt_32f`,
`volk_32f_reciprocal_32f`, `volk_32f_s32f_calc_spectral_noise_floor_32f`,
`volk_32f_s32f_clamppuppet_32f`, `volk_32f_stddev_and_mean_32f_x2`,
`volk_32f_tan_32f`, `volk_32f_tanh_32f`, `volk_32f_x2_dot_prod_32f`,
`volk_32f_x2_powpuppet_32f`

The 104 `impl`-mode FAILs are tail-remainder and adversarial-edge divergences
(the #87 capability): dot-product/accumulator reductions, index_max ties,
approximate-math kernels (atan, tan, expfast, invsqrt, reciprocal) at edge
values. Each needs per-kernel triage: real defect vs tolerance/edge-semantics
question — that adjudication belongs to the per-kernel fix tickets, not this
snapshot.

## Clean sweeps (and their caveats)

- **Input immutability (#90): zero findings.** No kernel writes its input
  buffers on this host.
- **Misaligned runs (#91): zero findings** — with an explicit coverage
  boundary: puppet kernels are skipped in misaligned mode by design (the
  longjmp safety argument requires no internal-allocation impls under the
  signal guard). The known rotator2 `u_avx512f` misaligned `phase_Ptr` fault
  (fixed on `dev/all-prs` by #101) is reachable only through its puppet, so it
  is structurally outside this mode's coverage — not evidence of absence.
- **Canary (#89): zero guard violations.** The 165 `partial` rows are the
  documented reduction/index-kernel pattern, dominated by fixed-size-output
  kernels (e.g. `volk_32f_x3_sum_of_poly_32f` across all impls). Note #98
  (harness modes under-seed sum_of_poly's parameter array) when triaging that
  family.

## Reproducing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=ON
cmake --build build -j"$(nproc)"
python3 tools/run_harness_triage.py --build-dir build --out triage.csv
```
