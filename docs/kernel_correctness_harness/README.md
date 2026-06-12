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
| `HARNESS_MISALIGNED=1` | misaligned `_u_`-variant runs: every unaligned impl on deliberately misaligned buffers, with scoped signal trapping (POSIX only) | #91 |
| `HARNESS_COMBINED_NC=1` | combined negative control (the ctest `qa_harness_negative_control`) | #92 |
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

### 3. Output canary + AddressSanitizer (#89)

- **Blind spot:** qa checks only `[0, num_points)`; a one-past (or far-past)
  over/under-write lands in pool slack and is invisible.
- **Escaped defect:** tail one-past over/under-run class (`30ab2b0`).
- **How:** each output buffer is its own malloc with leading/trailing sentinel
  guards; two runs with distinct sentinels flag guard writes (always a defect)
  and never-written in-bounds elements (`partial` — expected for
  reduction/index kernels with fixed-size outputs). ASan redzones bracket the
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
  POSIX-only (compiled out to an explicit skip on Windows). Puppet kernels are
  excluded by design (no internal-allocation impls under the signal guard) —
  defects reachable only through puppets are outside this mode's coverage.
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
`kernel,impl,mode,result,failed_vlens` — one row per kernel × impl. `mode` is
`ref`/`impl` (sweep), `canary`, `immutable`, or `misaligned`. `result` is
`ok`, `FAIL`, `partial` (canary unwritten), or `skip`; rows with impl `-` are
kernel-level (skips, or failures not attributable to one impl, e.g. an
exception mid-run). `failed_vlens` is space-separated.

`tools/run_harness_triage.py` drives the full catalog — one process per
(kernel, mode) via the argv filter, so a kernel that hard-aborts becomes an
`abort` finding row instead of killing the run — and merges everything into
one CSV. Runner-synthesized rows use the runner's mode label `sweep` for the
default mode (in-binary rows say `ref`/`impl`) and carry a detail string
instead of vlens in `failed_vlens`: `timeout`, `signal=<NAME>`, `exit=<N>`,
`runner-error: <msg>` (result `abort`), or `no-qa-entry` (result `skip`).

Checked-in snapshots (the per-kernel fix backlog):

- [`triage-2026-06-11.csv`](triage-2026-06-11.csv) +
  [`triage-2026-06-11-summary.md`](triage-2026-06-11-summary.md) — the epic-#85
  closeout snapshot: 154 kernels × 4 mode legs, 2,796 rows, 113 finding rows
  (112 FAIL across 22 kernels, plus 1 abort on a 23rd), headlined by the two
  live motivating defects.

Retention: latest snapshot only — a new snapshot replaces the old files
(history stays in git), so the directory does not accumulate dated CSVs.

## Platform notes

- **MSVC/Windows:** stdout muting uses `_dup`/`NUL` mappings; the misaligned
  mode compiles to an explicit skip (no POSIX signals); everything else —
  including the combined negative control ctest — runs.
- **ASan builds:** canary far-past demo is gated to ASan; misaligned mode needs
  the `ASAN_OPTIONS` above.
- **Default qa:** `volk_test_all` and the per-kernel `qa_volk_*` ctests are
  byte-for-byte unaffected by every capability here; all harness behavior is
  opt-in via the env toggles.

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
