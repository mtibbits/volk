#
# Copyright 2011-2013 Free Software Foundation, Inc.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

########################################################################
# Setup profiler
########################################################################

# POSIX_MEMALIGN: If we have to fall back to `posix_memalign`.
if(HAVE_POSIX_MEMALIGN)
    message(STATUS "Use `posix_memalign` for aligned malloc!")
    add_definitions(-DHAVE_POSIX_MEMALIGN)
endif(HAVE_POSIX_MEMALIGN)

# MAKE volk_profile
add_executable(
    volk_profile
    ${CMAKE_CURRENT_SOURCE_DIR}/volk_profile.cc ${PROJECT_SOURCE_DIR}/lib/qa_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/volk_option_helpers.cc)
target_compile_features(volk_profile PUBLIC cxx_std_17)

if(MSVC)
    target_include_directories(
        volk_profile PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cmake/msvc>)
endif(MSVC)

target_include_directories(
    volk_profile
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/lib>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(ENABLE_STATIC_LIBS)
    target_link_libraries(volk_profile PRIVATE volk_static fmt::fmt)
    set_target_properties(volk_profile PROPERTIES LINK_FLAGS "-static")
else()
    target_link_libraries(volk_profile PRIVATE volk fmt::fmt)
endif()

install(
    TARGETS volk_profile
    DESTINATION bin
    COMPONENT "volk")

# MAKE volk_ulp_profile
add_executable(
    volk_ulp_profile
    ${CMAKE_CURRENT_SOURCE_DIR}/volk_ulp_profile.cc
    ${PROJECT_SOURCE_DIR}/lib/qa_utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/volk_option_helpers.cc)
target_compile_features(volk_ulp_profile PUBLIC cxx_std_17)

if(MSVC)
    target_include_directories(
        volk_ulp_profile PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cmake/msvc>)
endif(MSVC)

target_include_directories(
    volk_ulp_profile
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/lib>
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lib>
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(ENABLE_STATIC_LIBS)
    target_link_libraries(volk_ulp_profile PRIVATE volk_static fmt::fmt)
    set_target_properties(volk_ulp_profile PROPERTIES LINK_FLAGS "-static")
else()
    target_link_libraries(volk_ulp_profile PRIVATE volk fmt::fmt)
endif()

install(
    TARGETS volk_ulp_profile
    DESTINATION bin
    COMPONENT "volk")

# MAKE volk-config-info
add_executable(volk-config-info volk-config-info.cc
                                ${CMAKE_CURRENT_SOURCE_DIR}/volk_option_helpers.cc)
target_compile_features(volk-config-info PUBLIC cxx_std_17)

if(ENABLE_STATIC_LIBS)
    target_link_libraries(volk-config-info volk_static)
    set_target_properties(volk-config-info PROPERTIES LINK_FLAGS "-static")
else()
    target_link_libraries(volk-config-info volk)
endif()

install(
    TARGETS volk-config-info
    DESTINATION bin
    COMPONENT "volk")

# Round-trip regression test for the volk_config reader/writer (issue #158).
# Drives the volk_profile binary; needs Python 3 (already required for codegen).
if(ENABLE_TESTING)
    set(volk_config_roundtrip_test
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_volk_config_roundtrip.py)
    add_test(
        NAME volk_config_roundtrip
        COMMAND ${PYTHON_EXECUTABLE} ${volk_config_roundtrip_test}
                --binary $<TARGET_FILE:volk_profile>)
    # The test seeds the config so the kernel is skipped (no profiling) and runs
    # in well under a second. Bound it so any future over-match that triggers real
    # profiling fails closed instead of hanging CI.
    set_tests_properties(volk_config_roundtrip PROPERTIES TIMEOUT 60)
endif(ENABLE_TESTING)

# Launch volk_profile if requested to do so
if(ENABLE_PROFILING)
    if(DEFINED VOLK_CONFIGPATH)
        set(VOLK_CONFIG_ARG "-p${VOLK_CONFIGPATH}")
        set(VOLK_CONFIG "${VOLK_CONFIGPATH}/volk_config")
    endif()

    add_custom_command(
        OUTPUT ${VOLK_CONFIG}
        COMMAND volk_profile "${VOLK_CONFIG_ARG}"
        DEPENDS volk_profile
        COMMENT "Launching profiler, this may take a few minutes...")
    add_custom_target(volk-profile-run ALL DEPENDS ${VOLK_CONFIG})

endif()
