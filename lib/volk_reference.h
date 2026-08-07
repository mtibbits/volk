/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/*
 * Independent double-precision reference oracles for the kernel-correctness
 * harness (#88). Each oracle computes a kernel's mathematically-defined output
 * in double precision from the SAME input buffers the impls consumed, narrowing
 * to the kernel's native float/complex output type only at the final store.
 * Because it is independent of the kernel's own code, it catches defects that
 * every impl shares (e.g. the swapped-atan2 in volk_32fc_s32f_power_32fc), which
 * the impl-vs-impl harness cannot.
 */

#ifndef INCLUDED_VOLK_REFERENCE_H
#define INCLUDED_VOLK_REFERENCE_H

#include <volk/volk_complex.h>
#include <string>
#include <vector>

// Computes the double-precision golden output for ONE kernel.
//   inputs  : the kernel's input buffers (post load_random_data), in signature
//             order, each already cast-able to the kernel's input element type.
//   outputs : the golden output buffer(s) to fill, in signature order.
//             REDUCTION oracles write only the contracted prefix (e.g.
//             outputs[0][0]); the harness zero-fills both sides so untouched
//             tails compare equal. A wrong-prefix oracle IS caught (fcompare
//             runs the full vlen, so it diverges from the impl's zero tail);
//             for kernels present in BOTH registries, keep the prefix count
//             consistent with the volk_buffer_roles.cc entry the canary side
//             consults. Membership itself diverges by design (#191): buffer
//             roles cover the whole fixed-output class, oracles stay opt-in.
//   scalar  : test_params.scalar() — real part carries an s32f scalar
//             (e.g. power exponent, atan2 normalizeFactor); full value for s32fc.
//   vlen    : the USER vlen (number of elements), not the twiddled buffer size.
typedef void (*volk_reference_fn)(const std::vector<const void*>& inputs,
                                  const std::vector<void*>& outputs,
                                  lv_32fc_t scalar,
                                  unsigned int vlen);

struct volk_reference_entry {
    const char* name;     // exact kernel name, e.g. "volk_32fc_s32f_power_32fc"
    volk_reference_fn fn; // double-precision oracle
    float tol;            // reference comparison tolerance
    bool absolute;        // absolute (true) vs relative (false) comparison
};

// The opt-in registry. Sparse by design: only registered kernels use reference
// mode; everything else falls back to impl-vs-impl. Grows over time.
const std::vector<volk_reference_entry>& volk_reference_registry();

// Returns the entry for `name`, or nullptr if the kernel is not registered.
const volk_reference_entry* volk_reference_lookup(const std::string& name);

#endif /* INCLUDED_VOLK_REFERENCE_H */
