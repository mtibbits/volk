/* -*- c++ -*- */
/*
 * Copyright 2011-2012 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef INCLUDED_VOLK_CONFIG_FIXED_H
#define INCLUDED_VOLK_CONFIG_FIXED_H

/* LV_<provides> aliases below share the parent arch's bit index by design:
 * the consolidated arch's CPUID check guarantees all <provides>'d sub-extensions
 * are present together, so OR'ing them in generated dispatch-table deps masks
 * (e.g. `(1 << LV_AVX512F) | (1 << LV_AVX512DQ)`) degenerates to a single-bit
 * test - which is the correct semantic, not a bug. */
%for i, arch in enumerate(archs):
#define LV_${arch.name.upper()} ${i}
%for prov in arch.provides:
#define LV_${prov.upper()} ${i}
%endfor
%endfor

#endif /*INCLUDED_VOLK_CONFIG_FIXED*/
