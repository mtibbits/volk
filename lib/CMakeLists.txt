#
# Copyright 2011-2012,2014,2018 Free Software Foundation, Inc.
#
# This file is part of VOLK.
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

########################################################################
# header file detection
########################################################################
include(CheckIncludeFile)
check_include_file(cpuid.h HAVE_CPUID_H)
if(HAVE_CPUID_H)
    add_definitions(-DHAVE_CPUID_H)
endif()

check_include_file(intrin.h HAVE_INTRIN_H)
if(HAVE_INTRIN_H)
    add_definitions(-DHAVE_INTRIN_H)
endif()

check_include_file(fenv.h HAVE_FENV_H)
if(HAVE_FENV_H)
    add_definitions(-DHAVE_FENV_H)
endif()

check_include_file(dlfcn.h HAVE_DLFCN_H)
if(HAVE_DLFCN_H)
    add_definitions(-DHAVE_DLFCN_H)
    list(APPEND volk_libraries ${CMAKE_DL_LIBS})
endif()

########################################################################
# Setup the compiler name
########################################################################
set(COMPILER_NAME ${CMAKE_C_COMPILER_ID})
if(MSVC) #its not set otherwise
    set(COMPILER_NAME MSVC)
endif()

# Assume "AppleClang == Clang".
string(TOLOWER ${COMPILER_NAME} COMPILER_NAME_LOWER)
string(REGEX MATCH "clang" COMPILER_NAME_LOWER ${COMPILER_NAME_LOWER})
if(${COMPILER_NAME_LOWER} MATCHES "clang")
    set(COMPILER_NAME "Clang")
endif()

message(STATUS "Compiler name: ${COMPILER_NAME}")

if(NOT DEFINED COMPILER_NAME)
    message(
        FATAL_ERROR "COMPILER_NAME undefined. Volk build may not support this compiler.")
endif()

########################################################################
# Special clang flag so flag checks can fail
########################################################################
if(COMPILER_NAME MATCHES "GNU")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-Werror=unused-command-line-argument"
                            HAVE_WERROR_UNUSED_CMD_LINE_ARG)
    if(HAVE_WERROR_UNUSED_CMD_LINE_ARG)
        set(VOLK_FLAG_CHECK_FLAGS "-Werror=unused-command-line-argument")
    endif()
endif()

########################################################################
# POSIX_MEMALIGN: If we have to fall back to `posix_memalign`,
# make it known to the compiler.
########################################################################
if(HAVE_POSIX_MEMALIGN)
    message(STATUS "Use `posix_memalign` for aligned malloc!")
    add_definitions(-DHAVE_POSIX_MEMALIGN)
endif(HAVE_POSIX_MEMALIGN)

########################################################################
# detect x86 flavor of CPU
########################################################################
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(i.86|x86|x86_64|amd64|AMD64)$")
    message(STATUS "x86* CPU detected")
    set(CPU_IS_x86 TRUE)
endif()

########################################################################
# determine passing architectures based on compile flag tests
########################################################################
execute_process(
    COMMAND
        ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
        ${PROJECT_SOURCE_DIR}/gen/volk_compile_utils.py --mode "arch_flags" --compiler
        "${COMPILER_NAME}"
    OUTPUT_VARIABLE arch_flag_lines
    OUTPUT_STRIP_TRAILING_WHITESPACE)

try_compile(HAVE_RVV_INTRINSICS ${PROJECT_BINARY_DIR}
            ${PROJECT_SOURCE_DIR}/cmake/Checks/check-rvv-intrinsics.c)
if(HAVE_RVV_INTRINSICS)
    message(STATUS "Checking RVV intrinsics - found")
else()
    message(STATUS "Checking RVV intrinsics - not found")
endif()

macro(check_arch arch_name)
    set(flags ${ARGN})
    set(have_${arch_name} TRUE)

    string(SUBSTRING "${arch_name}" 0 2 arch_prefix)
    foreach(flag ${flags})
        if(MSVC AND (${flag} STREQUAL "/arch:SSE2" OR ${flag} STREQUAL "/arch:SSE"))
            # SSE/SSE2 is supported in MSVC since VS 2005 but flag not available when compiling 64-bit so do not check
        elseif("${arch_prefix}" STREQUAL "rv" AND NOT HAVE_RVV_INTRINSICS)
            message(STATUS "Skipping ${arch_name} due to missing RVV intrinsics support")
            set(have_${arch_name} FALSE)
        else()
            include(CheckCXXCompilerFlag)
            set(have_flag have${flag})
            #make the have_flag have nice alphanum chars (just for looks/not necessary)
            execute_process(
                COMMAND ${PYTHON_EXECUTABLE} -c
                        "import re; print(re.sub(r'\\W', '_', '${have_flag}'))"
                OUTPUT_VARIABLE have_flag
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(VOLK_FLAG_CHECK_FLAGS)
                set(CMAKE_REQUIRED_FLAGS ${VOLK_FLAG_CHECK_FLAGS})
            endif()
            check_cxx_compiler_flag(${flag} ${have_flag})
            unset(CMAKE_REQUIRED_FLAGS)
            if(NOT ${have_flag})
                set(have_${arch_name} FALSE)
            endif()
        endif()
    endforeach()
    if(have_${arch_name})
        list(APPEND available_archs ${arch_name})
    endif()
endmacro(check_arch)

foreach(line ${arch_flag_lines})
    string(REGEX REPLACE "," ";" arch_flags ${line})
    check_arch(${arch_flags})
endforeach(line)

macro(OVERRULE_ARCH arch reason)
    message(STATUS "${reason}, Overruled arch ${arch}")
    list(REMOVE_ITEM available_archs ${arch})
endmacro(OVERRULE_ARCH)

macro(FORCE_ARCH arch reason)
    message(STATUS "${reason}, Forced arch ${arch}")
    list(APPEND available_archs ${arch})
endmacro(FORCE_ARCH)

########################################################################
# eliminate AVX on if not on x86, or if the compiler does not accept
# the xgetbv instruction, or {if not cross-compiling and the xgetbv
# executable does not function correctly}.
########################################################################
set(HAVE_AVX_CVTPI32_PS 0)
if(CPU_IS_x86)
    #########################################################################
    # eliminate AVX if cvtpi32_ps intrinsic fails like some versions of clang
    #########################################################################

    # check to see if the compiler/linker works with cvtpi32_ps instrinsic when using AVX
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        file(
            WRITE ${CMAKE_CURRENT_BINARY_DIR}/test_cvtpi32_ps.c
            "#include <immintrin.h>\nint main (void) {__m128 __a; __m64 __b; __m128 foo = _mm_cvtpi32_ps(__a, __b); return (0); }"
        )
        execute_process(
            COMMAND
                ${CMAKE_C_COMPILER} -mavx -o ${CMAKE_CURRENT_BINARY_DIR}/test_cvtpi32_ps
                ${CMAKE_CURRENT_BINARY_DIR}/test_cvtpi32_ps.c
            OUTPUT_QUIET ERROR_QUIET
            RESULT_VARIABLE avx_compile_result)
        if(NOT ${avx_compile_result} EQUAL 0)
            overrule_arch(avx "Compiler missing cvtpi32_ps instrinsic")
        else()
            set(HAVE_AVX_CVTPI32_PS 1)
        endif()
        file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/test_cvtpi32_ps
             ${CMAKE_CURRENT_BINARY_DIR}/test_cvtpi32_ps.c)
    else(CMAKE_SIZEOF_VOID_P EQUAL 4)
        # 64-bit compilations won't need this command so don't overrule AVX
        set(HAVE_AVX_CVTPI32_PS 0)
    endif(CMAKE_SIZEOF_VOID_P EQUAL 4)

    # Disable SSE4a if Clang is less than version 3.2
    if("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")
        # Figure out the version of Clang

        if(CMAKE_C_COMPILER_VERSION VERSION_LESS "3.2")
        endif(CMAKE_C_COMPILER_VERSION VERSION_LESS "3.2")
    endif("${CMAKE_C_COMPILER_ID}" MATCHES "Clang")

endif(CPU_IS_x86)

if(${HAVE_AVX_CVTPI32_PS})
    add_definitions(-DHAVE_AVX_CVTPI32_PS)
endif()

########################################################################
# if the CPU is not x86, eliminate all Intel SIMD
########################################################################

if(NOT CPU_IS_x86)
    overrule_arch(3dnow "Architecture is not x86 or x86_64")
    overrule_arch(mmx "Architecture is not x86 or x86_64")
    overrule_arch(sse "Architecture is not x86 or x86_64")
    overrule_arch(sse2 "Architecture is not x86 or x86_64")
    overrule_arch(sse3 "Architecture is not x86 or x86_64")
    overrule_arch(ssse3 "Architecture is not x86 or x86_64")
    overrule_arch(sse4_1 "Architecture is not x86 or x86_64")
    overrule_arch(sse4_2 "Architecture is not x86 or x86_64")
    overrule_arch(avx "Architecture is not x86 or x86_64")
    overrule_arch(avx512f "Architecture is not x86 or x86_64")
endif(NOT CPU_IS_x86)

########################################################################
# Select neon based on ARM ISA version
########################################################################

# First, compile a test program to see if compiler supports neon.

include(CheckCSourceCompiles)

# On native 32-bit armhf toolchains (e.g. gcc on a Raspberry Pi) NEON is not
# enabled by default, so the default-flag probe fails and -- without the fallback
# below -- every NEON arch is overruled and the build silently loses all NEON
# acceleration.
set(VOLK_NEON_TEST_SOURCE "#include <arm_neon.h>
int main(){ uint8_t *dest; uint8x8_t res; vst1_u8(dest, res); }")

# Flags that enable NEON on a native armhf toolchain, and a single ARM-target test.
# Both the fallback retry and the warning below must agree on "is this ARM", and the
# detection retry and the build must use the same flags -- keep them single-sourced.
set(VOLK_NEON_FALLBACK_FLAGS "-mfpu=neon -march=armv7-a")
set(VOLK_TARGET_IS_ARM FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|aarch64.*)$")
    set(VOLK_TARGET_IS_ARM TRUE)
endif()

set(CMAKE_C_FLAGS_SAVED "${CMAKE_C_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-uninitialized")

# First probe: default flags. Passes on aarch64 (mandatory ASIMD) and any
# toolchain with NEON enabled by default.
check_c_source_compiles("${VOLK_NEON_TEST_SOURCE}" neon_compile_result)

set(VOLK_NEON_ARMV7_FALLBACK FALSE)
if(NOT neon_compile_result AND VOLK_TARGET_IS_ARM)
    # Native armhf does not enable NEON by default. Retry with explicit
    # NEON-enabling flags before giving up. A DISTINCT result variable is required:
    # check_c_source_compiles caches its result, so reusing neon_compile_result
    # would return the stale failure.
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VOLK_NEON_FALLBACK_FLAGS}")
    check_c_source_compiles("${VOLK_NEON_TEST_SOURCE}" neon_compile_result_armv7)
    if(neon_compile_result_armv7)
        set(neon_compile_result TRUE)
        set(VOLK_NEON_ARMV7_FALLBACK TRUE)
    endif()
endif()

if(neon_compile_result)
    set(CMAKE_REQUIRED_INCLUDES ${PROJECT_SOURCE_DIR}/include)
    check_c_source_compiles("#include <volk/volk_common.h>
        int main(){__VOLK_ASM(\"vrev32.8 q0, q0\");}" have_neonv7_result)
    check_c_source_compiles("#include <volk/volk_common.h>
        int main(){__VOLK_ASM(\"sub v1.4s,v1.4s,v1.4s\");}" have_neonv8_result)

    if(NOT have_neonv7_result)
        overrule_arch(neonv7 "Compiler doesn't support neonv7")
    endif()

    if(NOT have_neonv8_result)
        overrule_arch(neonv8 "Compiler doesn't support neonv8")
    endif()
else(neon_compile_result)
    if(VOLK_TARGET_IS_ARM)
        message(WARNING
            "VOLK: NEON could not be enabled on ARM target "
            "'${CMAKE_SYSTEM_PROCESSOR}' even with ${VOLK_NEON_FALLBACK_FLAGS}. "
            "All NEON kernels are DISABLED; only generic C kernels will run. "
            "Check that the compiler supports NEON for this target.")
    endif()
    overrule_arch(neon "Compiler doesn't support NEON")
    overrule_arch(neonv7 "Compiler doesn't support NEON")
    overrule_arch(neonv8 "Compiler doesn't support NEON")
endif(neon_compile_result)

# Restore the probe-only flags (-Wno-uninitialized). When the armv7 fallback was
# needed to detect NEON, persist its flags into the build. A cross toolchain file
# would normally carry these; a native armhf build has none, so without this:
#   - the plain `neon` machine TU (compiled with the `neon` arch flags only, no
#     -mfpu=neon) fails to compile arm_neon.h intrinsics, and
#   - the hand-written NEON .s kernels (globbed in once neonv7 is available) fail to
#     assemble with "selected FPU does not support instruction".
# So propagate the flags to both the C compiler and the assembler.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_SAVED}")
if(VOLK_NEON_ARMV7_FALLBACK)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VOLK_NEON_FALLBACK_FLAGS}")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} ${VOLK_NEON_FALLBACK_FLAGS}")
endif()
########################################################################
# implement overruling in the ORC case,
# since ORC always passes flag detection
########################################################################
if(NOT ORC_FOUND)
    overrule_arch(orc "ORC support not found")
endif()

########################################################################
# implement overruling in the non-multilib case
# this makes things work when both -m32 and -m64 pass
########################################################################
if(NOT CROSSCOMPILE_MULTILIB AND CPU_IS_x86)
    include(CheckTypeSize)
    cmake_host_system_information(RESULT ASSUME_64BIT_HOST QUERY IS_64BIT)
    if(NOT ASSUME_64BIT_HOST)
        overrule_arch(64 "CPU width is 32 bits")
    endif()

    #MSVC 64 bit does not have MMX, overrule it
    if(MSVC)
        if(ASSUME_64BIT_HOST)
            overrule_arch(mmx "No MMX for Win64")
        endif()
        force_arch(sse "Built-in for MSVC > 2013")
        force_arch(sse2 "Built-in for MSVC > 2013")
    endif()

endif()

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "^riscv64$")
    message("---- Adding RISC-V ASM files")
    message("DEBUG: looking for ASM files in ${PROJECT_SOURCE_DIR}/kernels/volk/asm/riscv")
    include_directories(${PROJECT_SOURCE_DIR}/kernels/volk/asm/riscv)
    file(GLOB asm_files ${PROJECT_SOURCE_DIR}/kernels/volk/asm/riscv/*.s)
    foreach(asm_file ${asm_files})
        list(APPEND volk_sources ${asm_file})
        message(STATUS "Adding source file: ${asm_file}")
    endforeach(asm_file)
else()
    overrule_arch(riscv64 "machine is not riscv64")
endif()

########################################################################
# done overrules! print the result
########################################################################
message(STATUS "Available architectures: ${available_archs}")

########################################################################
# determine available machines given the available architectures
########################################################################
execute_process(
    COMMAND
        ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
        ${PROJECT_SOURCE_DIR}/gen/volk_compile_utils.py --mode "machines" --archs
        "${available_archs}"
    OUTPUT_VARIABLE available_machines
    OUTPUT_STRIP_TRAILING_WHITESPACE)

########################################################################
# Implement machine overruling for redundant machines:
# A machine is redundant when expansion rules occur,
# and the arch superset passes configuration checks.
# When this occurs, eliminate the redundant machines
# to avoid unnecessary compilation of subset machines.
########################################################################
foreach(arch mmx orc 64 32)
    foreach(machine_name ${available_machines})
        string(REPLACE "_${arch}" "" machine_name_no_arch ${machine_name})
        if(${machine_name} STREQUAL ${machine_name_no_arch})

        else()
            list(REMOVE_ITEM available_machines ${machine_name_no_arch})
        endif()
    endforeach(machine_name)
endforeach(arch)

########################################################################
# done overrules! print the result
########################################################################
message(STATUS "Available machines: ${available_machines}")

########################################################################
# Create rules to run the volk generator
########################################################################

#dependencies are all python, xml, and header implementation files
file(GLOB xml_files ${PROJECT_SOURCE_DIR}/gen/*.xml)
file(GLOB py_files ${PROJECT_SOURCE_DIR}/gen/*.py)
file(GLOB h_files ${PROJECT_SOURCE_DIR}/kernels/volk/*.h)

macro(gen_template tmpl output)
    list(APPEND volk_gen_sources ${output})
    add_custom_command(
        OUTPUT ${output}
        DEPENDS ${xml_files} ${py_files} ${h_files} ${tmpl}
        COMMAND
            ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
            ${PROJECT_SOURCE_DIR}/gen/volk_tmpl_utils.py --input ${tmpl} --output
            ${output} ${ARGN})
endmacro(gen_template)

make_directory(${PROJECT_BINARY_DIR}/include/volk)

gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk.tmpl.h
             ${PROJECT_BINARY_DIR}/include/volk/volk.h)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk.tmpl.c ${PROJECT_BINARY_DIR}/lib/volk.c)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_typedefs.tmpl.h
             ${PROJECT_BINARY_DIR}/include/volk/volk_typedefs.h)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_cpu.tmpl.h
             ${PROJECT_BINARY_DIR}/include/volk/volk_cpu.h)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_cpu.tmpl.c
             ${PROJECT_BINARY_DIR}/lib/volk_cpu.c)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_config_fixed.tmpl.h
             ${PROJECT_BINARY_DIR}/include/volk/volk_config_fixed.h)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_machines.tmpl.h
             ${PROJECT_BINARY_DIR}/lib/volk_machines.h)
gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_machines.tmpl.c
             ${PROJECT_BINARY_DIR}/lib/volk_machines.c)

set(BASE_CFLAGS NONE)
string(TOUPPER ${CMAKE_BUILD_TYPE} CBTU)
message(STATUS "BUILD TYPE = ${CBTU}")
message(STATUS "Base cflags = ${CMAKE_C_FLAGS_${CBTU}} ${CMAKE_C_FLAGS}")
set(COMPILER_INFO "")
if(MSVC)
    if(MSVC90) #Visual Studio 9
        set(cmake_c_compiler_version "Microsoft Visual Studio 9.0")
    elseif(MSVC10) #Visual Studio 10
        set(cmake_c_compiler_version "Microsoft Visual Studio 10.0")
    elseif(MSVC11) #Visual Studio 11
        set(cmake_c_compiler_version "Microsoft Visual Studio 11.0")
    elseif(MSVC12) #Visual Studio 12
        set(cmake_c_compiler_version "Microsoft Visual Studio 12.0")
    elseif(MSVC14) #Visual Studio 14
        set(cmake_c_compiler_version "Microsoft Visual Studio 14.0")
    endif()
else()
    execute_process(COMMAND ${CMAKE_C_COMPILER} --version
                    OUTPUT_VARIABLE cmake_c_compiler_version)
endif(MSVC)
set(COMPILER_INFO
    "${CMAKE_C_COMPILER}:::${CMAKE_C_FLAGS_${GRCBTU}} ${CMAKE_C_FLAGS}\n${CMAKE_CXX_COMPILER}:::${CMAKE_CXX_FLAGS_${GRCBTU}} ${CMAKE_CXX_FLAGS}\n"
)

foreach(machine_name ${available_machines})
    #generate machine source
    set(machine_source ${CMAKE_CURRENT_BINARY_DIR}/volk_machine_${machine_name}.c)
    gen_template(${PROJECT_SOURCE_DIR}/tmpl/volk_machine_xxx.tmpl.c ${machine_source}
                 ${machine_name})

    #determine machine flags
    execute_process(
        COMMAND
            ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
            ${PROJECT_SOURCE_DIR}/gen/volk_compile_utils.py --mode "machine_flags"
            --machine "${machine_name}" --compiler "${COMPILER_NAME}"
        OUTPUT_VARIABLE ${machine_name}_flags
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    message(
        STATUS
            "BUILD INFO ::: ${machine_name} ::: ${COMPILER_NAME} ::: ${CMAKE_C_FLAGS_${CBTU}} ${CMAKE_C_FLAGS} ${${machine_name}_flags}"
    )
    set(COMPILER_INFO
        "${COMPILER_INFO}${machine_name}:::${COMPILER_NAME}:::${CMAKE_C_FLAGS_${CBTU}} ${CMAKE_C_FLAGS} ${${machine_name}_flags}\n"
    )
    if(${machine_name}_flags AND NOT MSVC)
        set_source_files_properties(${machine_source}
                                    PROPERTIES COMPILE_FLAGS "${${machine_name}_flags}")
    endif()

    #add to available machine defs
    string(TOUPPER LV_MACHINE_${machine_name} machine_def)
    list(APPEND machine_defs ${machine_def})
endforeach(machine_name)

# Convert to a C string to compile and display properly
string(STRIP "${cmake_c_compiler_version}" cmake_c_compiler_version)
string(STRIP ${COMPILER_INFO} COMPILER_INFO)
message(STATUS "Compiler Version: ${cmake_c_compiler_version}")
string(REPLACE "\n" " \\n" cmake_c_compiler_version ${cmake_c_compiler_version})
string(REPLACE "\n" " \\n" COMPILER_INFO ${COMPILER_INFO})

########################################################################
# Handle ASM support
#  on by default, but let users turn it off
########################################################################
set(ASM_ARCHS_AVAILABLE "neonv7" "neonv8")

set(FULL_C_FLAGS "${CMAKE_C_FLAGS}" "${CMAKE_CXX_COMPILER_ARG1}")

# sort through a list of all architectures we have ASM for
# if we find one that matches our current system architecture
# set up the assembler flags and include the source files
foreach(ARCH ${ASM_ARCHS_AVAILABLE})
    string(REGEX MATCH "${ARCH}" ASM_ARCH "${available_archs}")
    if(ASM_ARCH STREQUAL "neonv7")
        message(STATUS "---- Adding ASM files") # we always use ATT syntax
        message(STATUS "-- Detected neon architecture; enabling ASM")
        # architecture specific assembler flags are now set in the cmake toolchain file
        # then add the files
        include_directories(${PROJECT_SOURCE_DIR}/kernels/volk/asm/neon)
        file(GLOB asm_files ${PROJECT_SOURCE_DIR}/kernels/volk/asm/neon/*.s)
        foreach(asm_file ${asm_files})
            list(APPEND volk_sources ${asm_file})
            message(STATUS "Adding source file: ${asm_file}")
        endforeach(asm_file)
    endif()
    enable_language(ASM)
    message(STATUS "c flags: ${FULL_C_FLAGS}")
    message(STATUS "asm flags: ${CMAKE_ASM_FLAGS}")
endforeach(ARCH)

########################################################################
# Handle orc support
########################################################################
if(ORC_FOUND)
    #setup orc library usage
    include_directories(${ORC_INCLUDE_DIRS})
    link_directories(${ORC_LIBRARY_DIRS})

    #setup orc functions
    file(GLOB orc_files ${PROJECT_SOURCE_DIR}/kernels/volk/asm/orc/*.orc)
    foreach(orc_file ${orc_files})

        #extract the name for the generated c source from the orc file
        get_filename_component(orc_file_name_we ${orc_file} NAME_WE)
        set(orcc_gen ${CMAKE_CURRENT_BINARY_DIR}/${orc_file_name_we}.c)

        #create a rule to generate the source and add to the list of sources
        add_custom_command(
            COMMAND ${ORCC_EXECUTABLE} --include math.h --implementation --include
                    volk/volk_complex.h -o ${orcc_gen} ${orc_file}
            DEPENDS ${orc_file}
            OUTPUT ${orcc_gen})
        list(APPEND volk_sources ${orcc_gen})

    endforeach(orc_file)
else()
    message(STATUS "Did not find liborc and orcc, disabling orc support...")
endif()

########################################################################
# Handle the generated constants
########################################################################

message(STATUS "Loading version ${VERSION} into constants...")

#double escape for windows backslash path separators
string(REPLACE "\\" "\\\\" prefix "${prefix}")
string(REPLACE "${PROJECT_SOURCE_DIR}" "$BUILD_DIR" COMPILER_INFO "${COMPILER_INFO}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/constants.c.in
               ${CMAKE_CURRENT_BINARY_DIR}/constants.c @ONLY)

list(APPEND volk_sources ${CMAKE_CURRENT_BINARY_DIR}/constants.c)

########################################################################
# Setup the volk sources list and library
########################################################################
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden -Wno-deprecated-declarations")
endif()

list(APPEND volk_sources ${CMAKE_CURRENT_SOURCE_DIR}/volk_prefs.c
     ${CMAKE_CURRENT_SOURCE_DIR}/volk_rank_archs.c
     ${CMAKE_CURRENT_SOURCE_DIR}/volk_malloc.c ${volk_gen_sources})

#set the machine definitions where applicable
set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/volk.c ${CMAKE_CURRENT_BINARY_DIR}/volk_machines.c
    PROPERTIES COMPILE_DEFINITIONS "${machine_defs}")

if(MSVC)
    #add compatibility includes for stdint types
    include_directories(${PROJECT_SOURCE_DIR}/cmake/msvc)
    add_definitions(-DHAVE_CONFIG_H)
    #compile the sources as C++ due to the lack of complex.h under MSVC
    set_source_files_properties(${volk_sources} PROPERTIES LANGUAGE CXX)
endif()

# Verify per-machine dispatch-table integrity after codegen, before
# compiling any machine .c file. See mtibbits/volk#58
# for the bug class this catches.
add_custom_target(volk_check_dispatch_tables
    COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_DASH_B}
            ${PROJECT_SOURCE_DIR}/cmake/check_dispatch_tables.py
            --source-dir ${PROJECT_SOURCE_DIR}
            --build-lib-dir ${PROJECT_BINARY_DIR}/lib
    DEPENDS ${volk_gen_sources}
    COMMENT "Checking per-machine impl dispatch-table integrity"
    VERBATIM
)

# Codegen-equivalence harness (mtibbits/volk#78): define the
# ADD_CODEGEN_EQUIVALENCE_TUPLE / FINALIZE_CODEGEN_EQUIVALENCE_HARNESS API
# before any tuple-declaration call sites. The harness asserts
# byte-identical-or-within-noise machine code between declared impl pairs at
# build time. See cmake/check_framework_codegen_README.md.
include(${PROJECT_SOURCE_DIR}/cmake/CodegenEquivalence.cmake)

#Create an object to reference Volk source and object files.
#
#NOTE: This object cannot be used to propagate include directories or
#library linkage, but we have to define them here for compiling to
#work. There are options starting with CMake 3.13 for using the OBJECT
#to propagate this information.
add_library(volk_obj OBJECT ${volk_sources})
add_dependencies(volk_obj volk_check_dispatch_tables)
target_include_directories(
    volk_obj
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/kernels>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(VOLK_CPU_FEATURES)
    set_source_files_properties(volk_cpu.c PROPERTIES COMPILE_DEFINITIONS
                                                      "VOLK_CPU_FEATURES=1")
    target_include_directories(
        volk_obj
        PRIVATE $<TARGET_PROPERTY:CpuFeatures::cpu_features,INTERFACE_INCLUDE_DIRECTORIES>
    )
endif()
target_compile_features(volk_obj PUBLIC c_std_17)

#Configure object target properties
set_target_properties(volk_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Disable complex math NaN/INFO range checking for performance
include(CheckCCompilerFlag)
check_c_compiler_flag(-fcx-limited-range HAVE_C_LIMITED_RANGE)
if(HAVE_C_LIMITED_RANGE)
    set_target_properties(volk_obj PROPERTIES COMPILE_FLAGS "-fcx-limited-range")
    check_c_compiler_flag(-Wno-unused-command-line-argument HAVE_WNO_UNUSED_CMD_LINE_ARG)
    if(HAVE_WNO_UNUSED_CMD_LINE_ARG)
        set_target_properties(volk_obj PROPERTIES COMPILE_FLAGS
                                                  "-Wno-unused-command-line-argument")
    endif(HAVE_WNO_UNUSED_CMD_LINE_ARG)
endif(HAVE_C_LIMITED_RANGE)

# Disable complex math NaN/INFO range checking for performance
include(CheckCCompilerFlag)
check_c_compiler_flag(-fcx-limited-range HAVE_C_LIMITED_RANGE)
if(HAVE_C_LIMITED_RANGE)
    set_target_properties(volk_obj PROPERTIES COMPILE_FLAGS "-fcx-limited-range")
    check_c_compiler_flag(-Wno-unused-command-line-argument HAVE_WNO_UNUSED_CMD_LINE_ARG)
    if(HAVE_WNO_UNUSED_CMD_LINE_ARG)
        set_target_properties(volk_obj PROPERTIES COMPILE_FLAGS
                                                  "-Wno-unused-command-line-argument")
    endif(HAVE_WNO_UNUSED_CMD_LINE_ARG)
endif(HAVE_C_LIMITED_RANGE)

#Add dynamic library
#
#NOTE: The PUBLIC include directories and library linkage will be
#propagated when this target is used to build other targets. Also, the
#PUBLIC library linkage and include directories INSTALL_INTERFACE will
#be used to create the target import information. Ordering of the
#include directories is taken as provided; it -might- matter, but
#probably doesn't.
add_library(volk SHARED $<TARGET_OBJECTS:volk_obj>)
target_link_libraries(volk PUBLIC ${volk_libraries})
if(VOLK_CPU_FEATURES)
    target_link_libraries(volk PRIVATE $<BUILD_INTERFACE:CpuFeatures::cpu_features>)
endif()
target_include_directories(
    volk
    PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/kernels>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PUBLIC $<INSTALL_INTERFACE:include>)

# Bootstrap codegen-equivalence assertion (mtibbits/volk#78 first consumer):
# compile a byte-for-byte copy of volk_32f_x2_add_32f_a_avx into a separate TU
# and assert the whole function is byte_identical to the in-header impl. Proves
# the harness extracts and compares disassembly correctly on a known-equal
# baseline, independent of any fusion-framework code.
#
# The base-avx machine TU name is detected from the configured machine list,
# NOT hardcoded: with ORC it is avx_64_mmx_orc, without it is avx_64_mmx, and
# matching "^avx_" excludes avx2_/avx512*. If no base-avx machine is built
# (e.g. a non-x86 target), the bootstrap tuple is skipped and the harness runs
# with zero tuples -- still valid (no false positives on a clean tree).
set(_cge_avx_machine "")
foreach(_cge_m ${available_machines})
    if(_cge_m MATCHES "^avx_")
        set(_cge_avx_machine "${_cge_m}")
        break()
    endif()
endforeach()
# Gate the bootstrap tuple to native 64-bit builds. The reference TU hardcodes
# -m64 (below) to mirror the 64-bit host machine TU, but a 32-bit / cross target
# that still builds an avx machine -- notably Android NDK "x86" (i686, 32-bit) --
# compiles the machine TU 32-bit while the reference TU is forced 64-bit, so the
# two genuinely diverge (addl/movl vs addq) and byte_identical fails on benign
# bitness, not a real codegen regression. The reference TU cannot faithfully
# mirror a 32-bit/cross machine TU, so skip the tuple there (zero tuples is a
# valid, false-positive-free configuration). mtibbits/volk#145.
set(_cge_bootstrap_enabled FALSE)
if(_cge_avx_machine AND (CMAKE_SIZEOF_VOID_P EQUAL 8))
    set(_cge_bootstrap_enabled TRUE)
endif()
if(_cge_bootstrap_enabled)
    # Optimisation flags are INHERITED from the build config (CMAKE_C_FLAGS_<cfg>),
    # not hardcoded, so this TU matches the avx machine TU under any build type
    # (Release -O3, Debug -O0, ...) rather than only Release. Only the avx
    # machine arch flags and PIC (to match the shared lib) are added here.
    add_library(volk_codegen_bootstrap_ref OBJECT
        ${PROJECT_SOURCE_DIR}/cmake/codegen_bootstrap_ref.c)
    set_target_properties(volk_codegen_bootstrap_ref PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_compile_options(volk_codegen_bootstrap_ref PRIVATE
        -fvisibility=hidden
        -m64 -mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mavx)

    ADD_CODEGEN_EQUIVALENCE_TUPLE(
        KERNEL    volk_32f_x2_add_32f
        ISA       avx
        ALIGNMENT a
        CRITERION byte_identical
        IMPL_A_SYMBOL    volk_32f_x2_add_32f_a_avx
        IMPL_A_MACHINE_O volk_machine_${_cge_avx_machine}.c.o
        IMPL_B_SYMBOL    volk_32f_x2_add_32f_a_avx_ref
        IMPL_B_MACHINE_O codegen_bootstrap_ref.c.o
        REQUIRE_MNEMONIC "^vaddps"
    )
elseif(_cge_avx_machine)
    message(STATUS "codegen-equivalence: base-avx machine present but build is "
                   "not 64-bit (sizeof(void*)=${CMAKE_SIZEOF_VOID_P}); bootstrap "
                   "tuple skipped -- the 64-bit reference TU cannot match a "
                   "32-bit/cross machine TU (e.g. Android NDK x86)")
else()
    message(STATUS "codegen-equivalence: no base-avx machine built; "
                   "bootstrap tuple skipped (harness runs with zero tuples)")
endif()

# Codegen-equivalence harness (mtibbits/volk#78): emit the manifest from the
# ADD_CODEGEN_EQUIVALENCE_TUPLE() calls above and create the check target, then
# make the shared lib depend on it so a regression in a framework
# instantiation's machine code breaks the build. With zero tuples declared the
# manifest is empty and the check is a no-op (zero false positives on
# origin/main). Mirrors the volk_check_dispatch_tables wiring (PR #59 / B15).
FINALIZE_CODEGEN_EQUIVALENCE_HARNESS()
if(_cge_bootstrap_enabled)
    add_dependencies(volk_check_framework_codegen volk_codegen_bootstrap_ref)
endif()
add_dependencies(volk volk_check_framework_codegen)

#Configure target properties
if(ORC_FOUND)
    target_link_libraries(volk PRIVATE ${ORC_LIBRARIES})
endif()
if(NOT MSVC)
    target_link_libraries(volk PUBLIC m)
endif()
set_target_properties(volk PROPERTIES VERSION ${VERSION})
set_target_properties(volk PROPERTIES SOVERSION ${SOVERSION})
set_target_properties(volk PROPERTIES DEFINE_SYMBOL "volk_EXPORTS")

#Install locations
install(
    TARGETS volk
    EXPORT VOLK-export
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT "volk_runtime" # .so file
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT "volk_devel" # .lib file
    RUNTIME DESTINATION bin COMPONENT "volk_runtime" # .dll file
)

#Configure static library
#
#NOTE: The PUBLIC include directories and library linkage will be
#propagated when this target is used to build other targets. Also, the
#PUBLIC library linkage and include directories INSTALL_INTERFACE will
#be used to create the target import information. Ordering of the
#include directories is taken as provided; it -might- matter, but
#probably doesn't.
if(ENABLE_STATIC_LIBS)
    add_library(volk_static STATIC $<TARGET_OBJECTS:volk_obj>)
    target_link_libraries(volk_static PUBLIC ${volk_libraries})
    if(VOLK_CPU_FEATURES)
        target_link_libraries(volk_static
                              PRIVATE $<BUILD_INTERFACE:CpuFeatures::cpu_features>)
    endif()
    if(ORC_FOUND)
        target_link_libraries(volk_static PUBLIC ${ORC_LIBRARIES_STATIC})
    endif()
    if(NOT ANDROID)
        target_link_libraries(volk_static PRIVATE pthread)
    endif()
    if(NOT MSVC)
        target_link_libraries(volk_static PUBLIC m)
    endif()
    target_include_directories(
        volk_static
        PUBLIC $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/kernels>
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC $<INSTALL_INTERFACE:include>)

    set_target_properties(volk_static PROPERTIES OUTPUT_NAME volk)

    install(
        TARGETS volk_static
        EXPORT VOLK-export
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT "volk_devel")
endif(ENABLE_STATIC_LIBS)

########################################################################
# Build the QA test application
########################################################################
if(ENABLE_TESTING)

    make_directory(${CMAKE_CURRENT_BINARY_DIR}/.unittest)
    include(VolkAddTest)
    if(ENABLE_STATIC_LIBS)
        volk_gen_test(volk_test_all SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testqa.cc
                      ${CMAKE_CURRENT_SOURCE_DIR}/qa_utils.cc TARGET_DEPS volk_static)
        volk_gen_test(
            volk_test_correctness SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/test_correctness.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/volk_reference.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/qa_canary_kernel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/qa_utils.cc TARGET_DEPS volk_static)
    else()
        volk_gen_test(volk_test_all SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/testqa.cc
                      ${CMAKE_CURRENT_SOURCE_DIR}/qa_utils.cc TARGET_DEPS volk)
        volk_gen_test(
            volk_test_correctness SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/test_correctness.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/volk_reference.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/qa_canary_kernel.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/qa_utils.cc TARGET_DEPS volk)
    endif()
    target_link_libraries(volk_test_all fmt::fmt)
    target_link_libraries(volk_test_correctness fmt::fmt)

    foreach(kernel ${h_files})
        get_filename_component(kernel ${kernel} NAME)
        string(REPLACE ".h" "" kernel ${kernel})
        volk_add_test(${kernel} volk_test_all)
    endforeach()

    # Combined negative control for the kernel-correctness harness (epic #85
    # closeout, #92): asserts the assembled harness still detects the two
    # escaped-defect classes that motivated it, each via its intended
    # capability. Plain ctest, every platform, no special build or env beyond
    # the mode toggle. If a change blinds the harness to either class, CI
    # breaks here.
    volk_add_test(harness_negative_control volk_test_correctness ENVIRONS
                  "HARNESS_COMBINED_NC=1")

endif(ENABLE_TESTING)
