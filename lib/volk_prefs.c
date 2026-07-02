/* -*- c++ -*- */
/*
 * Copyright 2011, 2012, 2015, 2016, 2019, 2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(_MSC_VER)
#include <io.h>
#define access _access
#define F_OK 0
#else
#include <unistd.h>
#endif
#include <volk/volk_prefs.h>

/* Prefer static const, but in C (non-MSVC builds) that is not a
   compile-time constant, making char path[CONFIG_PATH_MAX] a VLA.
   MSVC builds force this file to C++ (see lib/CMakeLists.txt), where
   static const would work, but Linux/GCC compiles it as C.  Use enum
   for a portable compile-time constant without resorting to #define. */
enum { CONFIG_PATH_MAX = 1024 };

void volk_get_config_path(char* path, bool read)
{
    if (!path)
        return;
    const char* suffix = "/.volk/volk_config";
    const char* suffix2 = "/volk/volk_config"; // non-hidden
    char* home = NULL;

    // allows config redirection via env variable
    home = getenv("VOLK_CONFIGPATH");
    if (home != NULL) {
        snprintf(path, CONFIG_PATH_MAX, "%s%s", home, suffix2);
        if (!read || access(path, F_OK) != -1) {
            return;
        }
    }

    // check for XDG_CONFIG_HOME (Linux/Unix)
    // XDG spec: "If $XDG_CONFIG_HOME is either not set or empty,
    // a default equal to $HOME/.config should be used."
    home = getenv("XDG_CONFIG_HOME");
    if (home != NULL && home[0] != '\0') {
        snprintf(path, CONFIG_PATH_MAX, "%s%s", home, suffix2);
        if (!read || access(path, F_OK) != -1) {
            return;
        }
    }

    // check for XDG default location ($HOME/.config/volk)
    home = getenv("HOME");
    if (home != NULL && home[0] != '\0') {
        snprintf(path, CONFIG_PATH_MAX, "%s/.config%s", home, suffix2);
        if (!read || access(path, F_OK) != -1) {
            return;
        }
    }

    // check for legacy user-local config file (read-only fallback)
    // Note: read-only so that new profiles write to XDG locations above.
    // volk_profile.cc migration logic depends on read and write paths
    // diverging here to detect legacy configs that should be migrated.
    home = getenv("HOME");
    if (home != NULL) {
        snprintf(path, CONFIG_PATH_MAX, "%s%s", home, suffix);
        if (read && (access(path, F_OK) != -1)) {
            return;
        }
    }

    // check for config file in APPDATA (Windows)
    home = getenv("APPDATA");
    if (home != NULL) {
        snprintf(path, CONFIG_PATH_MAX, "%s%s", home, suffix);
        if (!read || (access(path, F_OK) != -1)) {
            return;
        }
    }

    // check for system-wide config file
    if (access("/etc/volk/volk_config", F_OK) != -1) {
        strncpy(path, "/etc", 512);
        strcat(path, suffix2);
        if (!read || (access(path, F_OK) != -1)) {
            return;
        }
    }

    // If still no path was found set path[0] to '0' and fall through
    path[0] = 0;
    return;
}

// Reads one full line into *buf (caller-owned, must be free()d), returning its
// length or -1 at EOF/alloc-failure. Hand-rolled rather than POSIX getline()
// because this TU is compiled as C++ under MSVC, where getline() is absent. The
// 128 below is a growth seed, not a line-length cap: the buffer grows without
// limit, so a long line is parsed as one record instead of split (cf. line[512]).
static long read_config_line(char** buf, size_t* cap, FILE* f)
{
    size_t len = 0;
    int c;
    while ((c = getc(f)) != EOF) {
        if (len + 1 >= *cap) { // reserve one byte for the NUL terminator
            size_t ncap = *cap ? *cap * 2 : 128;
            char* new_buf = (char*)realloc(*buf, ncap);
            if (!new_buf) {
                printf("volk_load_preferences: bad malloc\n");
                return -1; // *buf still points at the valid old block
            }
            *buf = new_buf;
            *cap = ncap;
        }
        (*buf)[len++] = (char)c;
        if (c == '\n')
            break;
    }
    if (len == 0) // EOF with no bytes read
        return -1;
    (*buf)[len] = '\0';
    return (long)len;
}

size_t volk_load_preferences(volk_arch_pref_t** prefs_res)
{
    FILE* config_file;
    char path[CONFIG_PATH_MAX];
    char* line = NULL;
    size_t line_cap = 0;
    size_t n_arch_prefs = 0;
    volk_arch_pref_t* prefs = NULL;

    // get the config path
    volk_get_config_path(path, true);
    if (!path[0])
        return n_arch_prefs; // no prefs found
    config_file = fopen(path, "r");
    if (!config_file)
        return n_arch_prefs; // no prefs found

    // reset the file pointer and write the prefs into volk_arch_prefs
    while (read_config_line(&line, &line_cap, config_file) != -1) {
        void* new_prefs = realloc(prefs, (n_arch_prefs + 1) * sizeof(*prefs));
        if (!new_prefs) {
            printf("volk_load_preferences: bad malloc\n");
            break;
        }
        prefs = (volk_arch_pref_t*)new_prefs;
        volk_arch_pref_t* p = prefs + n_arch_prefs;
        if (sscanf(line, "%127s %127s %127s", p->name, p->impl_a, p->impl_u) == 3 &&
            !strncmp(p->name, "volk_", 5)) {
            n_arch_prefs++;
        }
    }
    free(line);
    fclose(config_file);
    *prefs_res = prefs;
    return n_arch_prefs;
}
