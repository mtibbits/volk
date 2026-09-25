/* -*- c++ -*- */
/*
 * Copyright 2011-2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

<%
    sd_machine = machine_dict[args[0]]
    sd_arch_names = sd_machine.arch_names
    arch_idx = {arch.name: i for i, arch in enumerate(archs)}

    def dep_score(impl):
        return sum(1 << arch_idx[d] for d in impl.deps) if impl.deps else 0

    def pick_best(impls, want_aligned):
        candidates = [i for i in impls if i.is_aligned == want_aligned]
        if not candidates:
            candidates = [i for i in impls if i.is_aligned != want_aligned]
        return max(candidates, key=dep_score) if candidates else None
%>

//! Prints a list of machines available
static inline void volk_list_machines(void)
{
    printf("${sd_machine.name}\n");
}

//! Returns the name of the machine this instance will use
static inline const char* volk_get_machine(void)
{
    return "${sd_machine.name}";
}

//! Get the machine alignment in bytes
static inline size_t volk_get_alignment(void)
{
    return ${sd_machine.alignment};
}

/*!
 * Is the pointer on the compile-time selected machine's alignment boundary?
 *
 * Unlike dynamic dispatch builds, static dispatch embeds the machine alignment
 * in this inline function, so no runtime initialization is required before
 * calling it.
 *
 * \param ptr the pointer to some memory buffer
 * \return 1 for alignment boundary, else 0
 */
static inline bool volk_is_aligned(const void* ptr)
{
    return ((intptr_t)(ptr) & (intptr_t)${sd_machine.alignment - 1}) == 0;
}

%for arch in sd_machine.archs:
#define LV_HAVE_${arch.name.upper()} 1
%endfor

/* Forward-declare all kernel dispatchers so that cross-kernel calls inside
 * implementation headers can resolve to the generic name without requiring
 * the full definition to already be visible. */
%for kern in kernels:
static inline void ${kern.name}(${kern.arglist_full});
%endfor

%for kern in kernels:
#include <volk/${kern.name}.h>
%endfor

/* Define _a/_u aliases and the alignment-dispatching inline for each kernel. */
%for kern in kernels:
<%
    impls = kern.get_impls(sd_arch_names)
    best_a = pick_best(impls, True)
    best_u = pick_best(impls, False)
%>\
#define ${kern.name}_a ${kern.name}_${best_a.name}
#define ${kern.name}_u ${kern.name}_${best_u.name}

static inline void ${kern.name}(${kern.arglist_full})
{
    if (volk_is_aligned(<% num_open_parens = 0 %>\
%for arg_type, arg_name in kern.args:
%if '*' in arg_type:
VOLK_OR_PTR(${arg_name},<% num_open_parens += 1 %>\
%endif
%endfor
0<% end_open_parens = ')'*num_open_parens %>${end_open_parens}))
        ${kern.name}_a(${kern.arglist_names});
    else
        ${kern.name}_u(${kern.arglist_names});
}

static inline void ${kern.name}_manual(${kern.arglist_full}, const char* impl_name)
{
    (void)impl_name;
    ${kern.name}(${kern.arglist_names});
}

static inline volk_func_desc_t ${kern.name}_get_func_desc(void)
{
%if best_a.name == best_u.name:
    static const char* impl_names[] = { "${best_a.name}" };
    static const int impl_deps[] = { 0 };
    static const bool impl_alignment[] = { true };
    static const volk_func_desc_t desc = { impl_names, impl_deps, impl_alignment, 1 };
%else:
    static const char* impl_names[] = { "${best_a.name}", "${best_u.name}" };
    static const int impl_deps[] = { 0, 0 };
    static const bool impl_alignment[] = { true, false };
    static const volk_func_desc_t desc = { impl_names, impl_deps, impl_alignment, 2 };
%endif
    return desc;
}

%endfor
