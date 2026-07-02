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
    set(one_value
        KERNEL ISA ALIGNMENT CRITERION
        IMPL_A_SYMBOL IMPL_A_MACHINE_O
        IMPL_B_SYMBOL IMPL_B_MACHINE_O)
    cmake_parse_arguments(CGE "" "${one_value}" "" ${ARGN})

    foreach(required IN LISTS one_value)
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

    set(frag "{\"kernel\":\"${CGE_KERNEL}\",\"isa\":\"${CGE_ISA}\",\"alignment\":\"${CGE_ALIGNMENT}\",\"criterion\":\"${CGE_CRITERION}\",\"impl_a\":{\"symbol\":\"${CGE_IMPL_A_SYMBOL}\",\"machine_o\":\"${CGE_IMPL_A_MACHINE_O}\"},\"impl_b\":{\"symbol\":\"${CGE_IMPL_B_SYMBOL}\",\"machine_o\":\"${CGE_IMPL_B_MACHINE_O}\"}}")
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
