# Kernel-correctness test harness (epic #85)

VOLK's default qa suite (`volk_test_all`) compares each SIMD impl against a
reference that can share the impl's assumptions, inspects only
`[0, num_points)`, exercises a single tail remainder, never runs on misaligned
buffers, and never checks input immutability. Each of those five structural
blind spots has let a real kernel defect ship or nearly ship. This harness
closes them with five opt-in detection capabilities that leave the default qa
suite byte-for-byte unchanged, plus a CI-enforced combined negative control
that proves the assembled harness still catches the defect classes that
motivated it.

A sixth blind spot is positional: fixed-position tie edges (qa's
`test_params_index_fc`, this harness's edge-value injection) place the first
tied element in an early-scanned lane, so an impl whose horizontal reduce
scans lanes out of memory order can violate the first-index-wins tie-break and
still pass every fixed edge (#195 — all four avx2 complex index variants did).
The standalone ctest `qa_index_tie_sweep` closes it by planting an
equal-magnitude extremum pair at every position pair (i, j) for vlens
8/32/37 and requiring every available impl to return the first tied index.

## How to run

Build with testing enabled, then run the harness binary directly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=ON
cmake --build build -j"$(nproc)"
LD_LIBRARY_PATH=build/lib build/lib/volk_test_correctness [kernel-name-filter]
```

`argv[1]` is an optional exact kernel-name filter. Modes are selected by
environment variable:

| env toggle | mode | child |
|---|---|---|
| (none) | tail-remainder sweep (vlens 1–40, 131071, 1000003) + adversarial edge values; kernels in the reference registry run against the independent double-precision oracle (`ref`), the rest impl-vs-impl (`impl`) | #87 / #88 |
| `HARNESS_CANARY=1` | output-buffer canary: guarded own-malloc buffers, two-sentinel over/under-write + unwritten-element checks | #89 |
| `HARNESS_CANARY_ASAN_DEMO=1` | (with `HARNESS_CANARY`, ASan build only) far-past over-run demo proving the guarded buffers are ASan-bracketed | #89 |
| `HARNESS_IMMUTABLE=1` | input-immutability canary: byte-exact post-run compare of every input against its pristine pre-image | #90 |
| `HARNESS_MISALIGNED=1` | misaligned `_u_`-variant runs: every unaligned impl on deliberately misaligned buffers, with scoped signal trapping (POSIX only); puppet kernels run under fork isolation with `(fork-isolated)`-tagged rows (ctest `qa_misaligned_puppet_control`); strict-UBSan regression detector is the ctest `qa_strict_misaligned_canary` (ASAN build type only) | #91 / #221 / #162 |
| `HARNESS_COMBINED_NC=1` | combined negative control (the ctest `qa_harness_negative_control`) | #92 |
| (none — standalone ctest `qa_index_tie_sweep`, binary `volk_test_index_tie_sweep`) | tie-position sweep for the complex index kernels: equal-magnitude extremum pair at every position pair (vlens 8/32/37), every available impl must return the FIRST tied index; avx2 coverage floor keyed to the hardware capability | #195 |
| `HARNESS_REPORT=path` | write the per-kernel × per-impl CSV (format below) in any mode | #87 / #92 |
| `HARNESS_VERBOSE=1` | unmute the per-impl stdout of the underlying qa runs | — |

## The five capabilities

Each capability closes one qa blind spot, motivated by a concrete escaped
defect. Every mode runs a built-in planted-kernel negative control before the
real sweep and exits 2 ("NEGATIVE CONTROL LOST") if its own detector is broken.

### 1. Tail-remainder sweep + adversarial edges (#87)

- **Blind spot:** qa runs one (prime) vlen — exactly one `num_points mod width`
  remainder per SIMD width — on smooth random data, so remainder-specific tail
  bugs and edge-case branches (clamps, saturation, zero/sign handling) are
  never exercised.
- **Escaped defects:** remainder-specific tail bugs (`aacf5d6`); the
  `volk_32f_tanh_32f` stale-input-pointer clamp branch (`e29bc7f`) — smooth
  data never enters the clamp.
- **How:** re-runs the existing comparison at every vlen 1–40 plus two large
  primes, with edge values (0, ±1, ±4.97, ±5, ±6, ±8 / complex equivalents)
  injected at the head of the test data.

### 2. Independent double-precision reference (#88)

- **Blind spot:** qa compares impls against a reference that can share their
  assumptions; a defect ALL impls share (including generic) is invisible.
- **Escaped defect:** `volk_32fc_s32f_power_32fc` swapped `atan2f` arguments +
  negated cosine (`e29bc7f`) — wrong for every non-trivial input, `ok` to
  impl-vs-impl because the generic IS the defect.
- **How:** an opt-in registry (`lib/volk_reference.cc`) of double-precision
  oracles computed from the same inputs; every impl (generic included) is
  compared against the oracle. Registered kernels report mode `ref`.

#### Reduction-tolerance methodology (#118 — the pattern the reduction family applies)

For reductions (dot products, accumulators, stddev, noise floor), impl-vs-generic
comparison judges the WRONG side: the serial generic's absolute error grows
~linearly in vlen (random-sign accumulation of per-step roundings) and measures
4–6× LARGER than the W-way SIMD partial sums'. Reductions therefore get a
reference-registry oracle, and TWO tolerances, each derived — never hand-bumped —
at ITS consumer's max vlen as `ceil_1sf(margin × max observed error)`, where the
margin is **2.5× for reductions** (the metric is a single random scalar per run —
extreme-value draws at fleet scale falsified a 1.5× first cut within hours) vs the
1.5× used for per-element max metrics (#173/#174, max-stable over the vector):

- **`ref.tol`** (the registry entry; consumed by this sweep at every vlen up to
  1000003): covers max BOTH-SIDES error vs the oracle at 1000003 — generic runs
  ref mode too.
- **`kp.tol`** (`lib/kernel_tests.h`; consumed by single-vlen testqa): covers the
  max impl-vs-generic delta at 131071.

The "max observed" must be TAIL-SAMPLED: at least **200 seeds for `kp.tol`
(131071) and 60 seeds for `ref.tol` (1000003)**, taking the maximum over BOTH
sides and ALL impls (generic included; the widest accumulation-order spread is
sometimes generic-vs-`block`, not generic-vs-SIMD). A 10-seed max undersamples
this single-random-scalar tail ~3× — measured across the #119–#123 family, where
a 10-seed first cut left two bounds below their observed 200/60-seed tails.

Both ABSOLUTE: zero-mean reductions cross zero, so relative bounds are ill-posed
near |result| → 0 (the #174 doctrine), and no single relative number is honest
across vlens anyway (relative error itself grows ~√vlen). Measurement recipe and
derivation: devDoc Issue-Fork-118/adjudication.md; per-kernel comments carry only
the kernel's numbers and cite this section.

### 3. Output canary + AddressSanitizer (#89)

- **Blind spot:** qa checks only `[0, num_points)`; a one-past (or far-past)
  over/under-write lands in pool slack and is invisible.
- **Escaped defect:** tail one-past over/under-run class (`30ab2b0`).
- **How:** each output buffer is its own malloc with leading/trailing sentinel
  guards; two runs with distinct sentinels flag guard writes (always a defect)
  and never-written in-bounds elements (`partial` for kernels not in the #161
  buffer-role registry). Kernels registered in the buffer-role registry — every
  fixed-output reduction/index kernel in the QA roster, as of #191 — instead get
  a hard `ok`/`FAIL` against their declared cardinality. ASan redzones bracket the
  guarded buffers for far-past coverage (demo gated to ASan builds).

### 4. Input-immutability canary (#90)

- **Blind spot:** nothing checks that a kernel leaves its read-only inputs
  untouched.
- **Escaped defect class:** a kernel scribbling on its input corrupts the
  caller's data flow downstream.
- **How:** byte-exact (`memcmp`, not a hash) post-run comparison of every input
  buffer against a pristine pre-image the kernel never receives.

### 5. Misaligned `_u_`-variant runs (#91)

- **Blind spot:** qa runs only on `volk_get_alignment()`-aligned buffers, so an
  unaligned impl that secretly assumes alignment (movaps-in-a-`_u_`-kernel)
  never faults in qa.
- **Escaped defect class:** `_u_` variants using aligned loads/stores — crash
  on real-world misaligned data.
- **How:** every impl whose dispatch metadata says unaligned runs on
  element-aligned but vector-misaligned buffers; SIGSEGV/SIGBUS/SIGILL are
  trapped with a scoped handler (one crashing impl = one recorded FAIL, the
  run continues) and output is compared against the same impl's aligned run.
  POSIX-only (compiled out to an explicit skip on Windows).
- **Puppet kernels (#162):** covered via **fork isolation**, not the signal
  guard — puppets may allocate internally (3 of the 15 registered puppets do,
  incl. encodepolar's 7 `volk_malloc` sites), which would make
  `longjmp`-under-signal-guard recovery unsound. Each puppet impl's whole
  aligned/misaligned/compare cycle runs in a forked child; a fault kills only
  the child and the parent records it from the wait status + a phase pipe
  (fail-closed: an impl whose fork setup fails is not counted, so an
  all-failed kernel reports `skip (fork setup failed: N impl-runs)`, never
  `ok`; partial losses print a `setup-failed impl-runs: N` qualifier on the
  row). Puppet rows carry a `(fork-isolated)` stdout tag derived from the
  summary the fork branch itself reports — the tag observes the routing that
  happened, not the predicate that selects it. One policy exclusion remains:
  `volk_8u_conv_k7_r2puppet_8u` prints `skip … (excluded: #96 conv_k7)`
  until its decision-buffer overflow (#96) lands — the corruption is
  child-contained but would hard-red every ASan lane for a known, tracked
  defect. A fork-path negative-control **trio** (planted ok, fault, and
  alignment-sensitive-diverging kernels routed through forked children) runs
  in **every** misaligned invocation, including the
  `qa_strict_misaligned_canary` lane; losing any of the three verdict
  transports (ok / crash / diverged) aborts with exit 2. The fault twin is
  the regression proof for the #101 defect class (an unaligned fault in a
  puppet-only worker), pinned by the ctest `qa_misaligned_puppet_control`,
  whose pass/fail regexes require the rotator2 puppet's fork-routed `ok` row
  and forbid FAIL rows, NC loss, or setup-failed qualifiers ("covered AND
  clean"). That ctest registers by default on **Linux x86_64 only** (the
  validated platform; CMake option `VOLK_MISALIGNED_PUPPET_CTEST`, opt-in
  elsewhere) and never on static-dispatch builds, where
  `volk_get_alignment()` is 1 and the mode's degenerate-alignment guard
  fails closed by design — broadening to other arches/OSes after a
  cross-platform probe is a tracked follow-up.
- **Mapping-completeness note (#162):** for each puppet the sweep compares the
  master kernel's impl-name list against the puppet's wrappers and prints a
  stderr `note` for any master impl with no same-named wrapper — a stale
  puppet silently caps coverage for its whole kernel class (the
  gnuradio/volk#570 popcnt lesson). Advisory, name-set only: a wrapper that
  exists but calls the *wrong* master impl is a source-level property this
  check cannot see. The computation is self-checked at mode start against
  hand-built descs (must fire on a planted gap AND stay quiet on a complete
  pair; any misclassification is NC-LOST exit 2).
- **ASan note:** run with
  `ASAN_OPTIONS=handle_segv=0:handle_sigbus=0:handle_sigill=0:allow_user_segv_handler=1`
  or ASan's handler wins and aborts on the first planted fault.

## Combined negative control (#92)

The ctest `qa_harness_negative_control` (plain ctest, every platform, enforced
by the existing CI `ctest` runs) re-runs both motivating escaped defects as
**planted copies** in `lib/qa_canary_kernel.cc` — the live kernels get fixed by
the per-kernel campaign; the copies are permanent — and asserts six things:

1. the planted power defect IS flagged by the independent reference (#88);
2. the planted power defect is NOT flagged by impl-vs-impl (the blind-spot
   premise still holds — detection is for the right reason);
3. the corrected power twin passes reference mode ("reverting the
   reintroduction makes the harness pass", continuously enforced);
4. the planted stale-pointer tanh defect IS flagged by the tail/edge sweep
   (#87);
5. the same defect is NOT flagged without edge injection (reason-specificity:
   the edge capability is what catches it);
6. the corrected tanh twin passes the sweep.

Any deviation exits 2 with a `NEGATIVE CONTROL LOST` diagnosis. A wrong-reason
crash cannot score green: each assertion checks the specific capability's
verdict, not the process exit code. If a future change blinds the harness to
either defect class, CI breaks here.

## Triage report format and snapshots

With `HARNESS_REPORT=path` every mode writes
`kernel,impl,mode,result,failed_vlens,max_err` — one row per kernel × impl,
preceded by a `# volk-harness-report v2` version-marker line (`#`-prefixed so
readers skip it). `mode` is `ref`/`impl` (sweep), `canary`, `immutable`, or
`misaligned`. `result` is `ok`, `FAIL`, `partial` (canary unwritten), or `skip`;
rows with impl `-` are kernel-level (skips, or failures not attributable to one
impl, e.g. an exception mid-run). `failed_vlens` is space-separated.

`max_err` (`#135`) is the per-impl worst-case divergence magnitude across all
swept vlens — formatted `%.3g` (e.g. `4e+04`, `1.2e-06`, `inf`) — so a triager
can rank `FAIL`s by severity instead of treating a `4e+04` and a `1.2e-06` error
as the same bare `FAIL`. It is populated for the `ref`/`impl` sweep and **empty**
for `canary`/`immutable`/`misaligned` (no numeric divergence) and for every
kernel-level `-` row (including ref `skip`). Note: an `ok` row still carries its
worst *within-tolerance* divergence (it is the largest observed, not zero), so
rank on `result == FAIL` first. A literal `0` means zero observed divergence,
not necessarily the baseline — a swept impl bit-identical to `generic` also
reads `0` (distinct from the *empty* field on skip / non-sweep rows). In `impl`
mode `generic` is the baseline and reads `0`; in `ref` mode `generic` is itself
compared against the oracle, so its `max_err` is a real value.

Reference mode (`ref`) reports `skip` — **not** a silent `ok` — when the oracle
cannot evaluate a registered kernel's shape (`#133`): a
`<kernel>,-,ref,skip,<reason>,` row (empty `max_err`) whose `failed_vlens` column
carries the reason (`unsupported-scalar-signature`, `unsupported-arity`, or
`ref-setup-failed`) instead of vlens. This makes a kernel the oracle could not
run distinguishable from one it ran and passed — important as the reference
registry grows (each Class-B adjudication adds an oracle, and a newly-registered
unsupported shape must surface as a coverage gap, not a false `ok`). The CI
combined negative control asserts this path stays a `skip`.

`tools/run_harness_triage.py` drives the full catalog — one process per
(kernel, mode) via the argv filter, so a kernel that hard-aborts becomes an
`abort` finding row instead of killing the run — and merges everything into
one CSV. Runner-synthesized rows use the runner's mode label `sweep` for the
default mode (in-binary rows say `ref`/`impl`) and carry a detail string
instead of vlens in `failed_vlens`: `timeout`, `signal=<NAME>`, `exit=<N>`,
`runner-error: <msg>` (result `abort`), or `no-qa-entry` (result `skip`).
`max_err` is empty on all runner-synthesized rows.

Checked-in snapshots (the per-kernel fix backlog):

- [`triage-2026-06-14.csv`](triage-2026-06-14.csv) +
  [`triage-2026-06-14-summary.md`](triage-2026-06-14-summary.md) — the seeded
  (`HARNESS_SEED=20260614`), severity-ranked v2 snapshot taken after the WP 2.1
  gate completed the harness (#106 Phase A per-kernel params, #133 ref-skip,
  #134 seed, #135 `max_err`): 2,854 rows, **46 finding rows** (down from the old
  113 — Phase A cleared ~67 rows of harness artifact). The summary's 10-seed
  stability study is the load-bearing classification: 9 stable genuine findings,
  6 reduction-family margin-flickers (→ #118), 8 harness-artifact clears. Triage
  by that table, not the single-seed CSV.

Retention: latest snapshot only — a new snapshot replaces the old files
(history stays in git), so the directory does not accumulate dated CSVs.

## Platform notes

- **MSVC/Windows:** stdout muting uses `_dup`/`NUL` mappings; the misaligned
  mode compiles to an explicit skip (no POSIX signals); everything else —
  including the combined negative control ctest — runs.
- **ASan builds:** canary far-past demo is gated to ASan; misaligned mode needs
  the `ASAN_OPTIONS` above. The control-kernel TU (`qa_canary_kernel.cc`) is
  compiled with `-fno-sanitize=alignment` on **every** non-Windows build
  (#221, gate widened by #162) so any UBSan-enabled lane (`halt_on_error=1`)
  can run the misaligned mode — the planted `movaps` still faults; only the
  pre-fault UBSan report is suppressed, and only for that one test-only TU.
  The flag is an accepted no-op when no sanitizer is enabled, and UBSan can
  arrive via injected `-fsanitize=undefined` flags on any build type (e.g.
  `-DCMAKE_BUILD_TYPE=Debug` sanitizer sweeps), not just `CBTU=ASAN` — which
  is why the exemption no longer follows the build type. Only the
  `qa_strict_misaligned_canary` ctest *registration* remains ASAN-gated; it
  runs the strict misaligned mode and goes red if the exemption is ever lost.
- **Default qa:** `volk_test_all` and the per-kernel `qa_volk_*` ctests are
  byte-for-byte unaffected by every capability here; all harness behavior is
  opt-in via the env toggles. The `qa_index_tie_sweep` ctest (#195) is
  standalone and unconditional: plain ctest, every platform, no env toggle,
  no special build type.

## Reproducibility (`HARNESS_SEED`)

`tools/run_harness_triage.py --seed N` (or an exported `HARNESS_SEED`)
pins the qa input data: the runner derives a distinct deterministic
seed per (kernel, mode) child process (`crc32`-mixed), and
`load_random_data` then draws from one per-process seeded stream, so
identical seeds reproduce identical rows — including the
`failed_vlens` lists that otherwise flicker at tolerance margins.
Unset (the default, including everything ctest runs) the data is
re-randomized per run, exactly as before. Because `load_random_data`
is shared, an exported `HARNESS_SEED` also pins default-qa
(`volk_test_all`) and `volk_profile` input data — a side effect, by
design. Snapshot summaries must record the seed used. Caveat: kernels
that read memory outside their seeded buffers (e.g. the #98
`sum_of_poly` defect) can still vary between identical-seed runs —
that variance is itself evidence of an out-of-bounds read.
(The shared `load_random_data` also serves `volk_ulp_profile` and the
per-kernel `qa_volk_*` ctests: an exported seed pins those too, and
in multi-kernel loops the per-process stream makes results
deterministic but iteration-order-coupled.)
