#
# Copyright 2026 Free Software Foundation, Inc.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# CodegenEquivalence.cmake
#
# Build-time codegen-equivalence harness for the volk fusion framework
# (mtibbits/volk#78). Provides ADD_CODEGEN_EQUIVALENCE_TUPLE() to declare
# (kernel, ISA, alignment, impl_a, impl_b, criterion) tuples that
# cmake/check_framework_codegen.py asserts after volk_obj compiles, and
# FINALIZE_CODEGEN_EQUIVALENCE_HARNESS() to emit the manifest + custom target.
#
# Mirrors the build-time-check pattern of the dispatch-table integrity check
# (mtibbits/volk#58, PR #59). See cmake/check_framework_codegen_README.md for
# the manifest format and the whole-function comparison contract.

# Accumulates per-tuple JSON fragments into a global property the finalize
# step stitches into a manifest. Use a CMake list (not APPEND_STRING) so the
# embedded double-quotes in each fragment survive intact.
set_property(GLOBAL PROPERTY CODEGEN_EQUIVALENCE_TUPLES "")

function(ADD_CODEGEN_EQUIVALENCE_TUPLE)
    set(required_args
        KERNEL ISA ALIGNMENT CRITERION
        IMPL_A_SYMBOL IMPL_A_MACHINE_O
        IMPL_B_SYMBOL IMPL_B_MACHINE_O)
    # REQUIRE_MNEMONIC is OPTIONAL: parsed but NOT in required_args, so existing
    # call sites without it are unaffected. When set, the harness additionally
    # asserts the compared function contains >=1 instruction whose mnemonic
    # matches the regex (scalar-fallback guard).
    #
    # REQUIRE_STANDALONE is an OPTIONAL boolean flag (orthogonal axis): pass the
    # bare keyword (no value) for a tuple whose dispatch relies on the impl
    # existing as a real, separately-dispatchable symbol. When set, the harness
    # treats a not-found label (SymbolNotEmittedError -- genuinely inlined
    # away, or absent) as a hard failure instead of skip-with-warning.
    # Omitted -> today's skip-with-warning behavior (subject to the
    # aggregate zero-coverage guard, mtibbits/volk#165).
    cmake_parse_arguments(CGE
        "REQUIRE_STANDALONE" "${required_args};REQUIRE_MNEMONIC" "" ${ARGN})

    foreach(required IN LISTS required_args)
        if(NOT DEFINED CGE_${required})
            message(FATAL_ERROR
                "ADD_CODEGEN_EQUIVALENCE_TUPLE: missing required arg ${required}")
        endif()
    endforeach()
    if(NOT CGE_CRITERION MATCHES "^(byte_identical|within_noise)$")
        message(FATAL_ERROR
            "ADD_CODEGEN_EQUIVALENCE_TUPLE: CRITERION must be byte_identical "
            "or within_noise (got '${CGE_CRITERION}')")
    endif()

    # Emit optional fields only when set, so tuples without them produce
    # byte-identical JSON to before this change. require_standalone is a flag, so
    # it is ALWAYS defined (TRUE/FALSE) -- gate on its truthiness, not DEFINED.
    set(_extra "")
    if(DEFINED CGE_REQUIRE_MNEMONIC)
        set(_extra ",\"require_mnemonic\":\"${CGE_REQUIRE_MNEMONIC}\"")
    endif()
    if(CGE_REQUIRE_STANDALONE)
        string(APPEND _extra ",\"require_standalone\":true")
    endif()

    set(frag "{\"kernel\":\"${CGE_KERNEL}\",\"isa\":\"${CGE_ISA}\",\"alignment\":\"${CGE_ALIGNMENT}\",\"criterion\":\"${CGE_CRITERION}\",\"impl_a\":{\"symbol\":\"${CGE_IMPL_A_SYMBOL}\",\"machine_o\":\"${CGE_IMPL_A_MACHINE_O}\"},\"impl_b\":{\"symbol\":\"${CGE_IMPL_B_SYMBOL}\",\"machine_o\":\"${CGE_IMPL_B_MACHINE_O}\"}${_extra}}")
    set_property(GLOBAL APPEND PROPERTY CODEGEN_EQUIVALENCE_TUPLES "${frag}")
endfunction()

# Emit the manifest JSON and the check target. Call once from
# lib/CMakeLists.txt AFTER all ADD_CODEGEN_EQUIVALENCE_TUPLE() calls and AFTER
# the volk_obj target is defined.
function(FINALIZE_CODEGEN_EQUIVALENCE_HARNESS)
    get_property(frags GLOBAL PROPERTY CODEGEN_EQUIVALENCE_TUPLES)
    string(JOIN "," frags_joined ${frags})
    set(manifest_path "${PROJECT_BINARY_DIR}/codegen_equivalence_manifest.json")
    file(WRITE "${manifest_path}" "{\"tuples\":[${frags_joined}]}\n")

    # llvm-objdump (with GNU objdump as a documented alternative) is used by
    # the script; resolve it if available so the target fails clearly when no
    # disassembler exists rather than mid-run.
    find_program(CGE_OBJDUMP NAMES llvm-objdump llvm-objdump-18 objdump)
    if(NOT CGE_OBJDUMP)
        message(WARNING
            "CodegenEquivalence: no llvm-objdump/objdump found; the "
            "codegen-equivalence check will error if any tuple is declared.")
        set(CGE_OBJDUMP "llvm-objdump")
    endif()

    add_custom_target(volk_check_framework_codegen
        COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
                ${PROJECT_SOURCE_DIR}/cmake/check_framework_codegen.py
                --manifest "${manifest_path}"
                --build-lib-dir "${PROJECT_BINARY_DIR}/lib/CMakeFiles"
                --objdump "${CGE_OBJDUMP}"
        DEPENDS volk_obj
        COMMENT "Checking codegen-equivalence between declared impl pairs"
        VERBATIM
    )
endfunction()
