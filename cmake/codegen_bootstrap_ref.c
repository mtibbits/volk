/* -*- c++ -*- */
/*
 * Copyright 2026 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
/*
 * Bootstrap codegen-equivalence reference for mtibbits/volk#78.
 *
 * Defines volk_32f_x2_add_32f_a_avx_ref as a byte-for-byte copy of the
 * existing _a_avx impl body from kernels/volk/volk_32f_x2_add_32f.h. This TU
 * is compiled with the SAME flags the impl gets in its machine TU
 * (-O3 -mavx), so the codegen-equivalence harness should find the two
 * functions byte_identical. This is the harness's first consumer: it proves
 * the harness extracts and compares whole-function disassembly correctly on a
 * known-equal baseline, independent of any fusion-framework code.
 *
 * No source labels are used -- the harness compares the whole function body.
 * Inserting labels would perturb codegen (a label symbol becomes a basic-block
 * boundary the optimizer must respect); whole-function comparison avoids that
 * entirely. See cmake/check_framework_codegen_README.md.
 *
 * The body below is intentionally identical to the in-header _a_avx impl. The
 * duplication is the experimental setup: two independent compilations of the
 * same logic, asserted equal. volk_32f_x2_add_32f_a_avx is mathematically
 * c = a + b over 8 AVX lanes and does not change, so the maintenance cost of
 * the copy is effectively zero. Child mtibbits/volk#79 brings the real
 * framework-instantiated impls online as the first non-bootstrap consumers.
 */
#include <immintrin.h>

void volk_32f_x2_add_32f_a_avx_ref(float* cVector,
                                   const float* aVector,
                                   const float* bVector,
                                   unsigned int num_points)
{
    unsigned int number = 0;
    const unsigned int eighthPoints = num_points / 8;

    float* cPtr = cVector;
    const float* aPtr = aVector;
    const float* bPtr = bVector;

    __m256 aVal, bVal, cVal;
    for (; number < eighthPoints; number++) {

        aVal = _mm256_load_ps(aPtr);
        bVal = _mm256_load_ps(bPtr);

        cVal = _mm256_add_ps(aVal, bVal);

        _mm256_store_ps(cPtr, cVal); // Store the results back into the C container

        aPtr += 8;
        bPtr += 8;
        cPtr += 8;
    }

    number = eighthPoints * 8;
    for (; number < num_points; number++) {
        *cPtr++ = (*aPtr++) + (*bPtr++);
    }
}
