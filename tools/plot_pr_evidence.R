#!/usr/bin/env Rscript
#
# Copyright 2026 Matthew Tibbits.
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Reproducible performance-PR evidence plots from volk_profile
# --trial-csv output. See mtibbits/volk#46 for design rationale.
#
# Usage:
#   Rscript tools/plot_pr_evidence.R <run1.csv> [<run2.csv> ...] \
#       [--outdir=DIR] [--kernels=k1,k2,...] [--labels=l1,l2,...]

# Deterministic byte-order arch sort across systems with different
# default locales (en_US vs de_DE produce different orderings for
# mixed-case strings).
invisible(Sys.setlocale("LC_COLLATE", "C"))

REQUIRED_PACKAGES <- c("ggplot2", "dplyr", "digest")
missing_pkgs <- REQUIRED_PACKAGES[
    !vapply(REQUIRED_PACKAGES, requireNamespace,
            logical(1), quietly = TRUE)
]
if (length(missing_pkgs) > 0) {
    install_cmd <- sprintf(
        "install.packages(c(%s))",
        paste(sprintf('"%s"', missing_pkgs), collapse = ", ")
    )
    cat("error: missing required R package(s): ",
        paste(missing_pkgs, collapse = ", "), "\n",
        "       run: ", install_cmd, "\n", sep = "",
        file = stderr())
    quit(status = 1, save = "no")
}

# ---- Constants ---------------------------------------------------

EXPECTED_CSV_HEADER <- c("kernel", "arch", "trial", "time_ms")
MAX_RUNS <- 8L
CANVAS_WIDTH_IN  <- 12
CANVAS_HEIGHT_IN <- 8
CANVAS_DPI       <- 150

# ColorBrewer "Dark2", hex values from colorbrewer2.org. Hard-coded
# rather than loaded via scale_fill_brewer() so a future ColorBrewer
# update cannot silently change our PR plots.
PALETTE_DARK2 <- c(
    "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
    "#66A61E", "#E6AB02", "#A6761D", "#666666"
)

# Order archs for plotting: `generic` always leftmost; other
# prefix-less algorithm names (generic_branchless, lut, polynomial,
# 1972magic, etc.) next, alphabetically; then SIMD archs grouped by
# family with a_/u_ pairs adjacent (a_ before u_ within each family).
# Within tier 2, the family sort key replaces '_' with ' ' so that
# variants (avx_fma, avx2_fma, sse4_1) sort *adjacent to their base*
# rather than after it lexicographically — pure byte order would
# place avx_fma after avx512f because '_' (0x5F) > '5' (0x35).
sort_archs_for_plot <- function(archs) {
    has_au_prefix <- grepl("^[au]_", archs)
    is_generic    <- archs == "generic"
    tier   <- ifelse(is_generic, 0L,
                     ifelse(!has_au_prefix, 1L, 2L))
    family <- ifelse(has_au_prefix, sub("^[au]_", "", archs), archs)
    family_key <- gsub("_", " ", family, fixed = TRUE)
    au     <- ifelse(has_au_prefix, substr(archs, 1L, 1L), "")
    archs[order(tier, family_key, au, archs)]
}

# ---- Error helpers ----------------------------------------------

stop_loud <- function(msg) {
    cat("error: ", msg, "\n", sep = "", file = stderr())
    quit(status = 1, save = "no")
}

stop_with_usage <- function(msg) {
    cat("error: ", msg, "\n", sep = "", file = stderr())
    print_usage()
    quit(status = 1, save = "no")
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---- Usage -------------------------------------------------------

print_usage <- function() {
    cat(
        "Usage: Rscript tools/plot_pr_evidence.R \\\n",
        "         <run1.csv> [<run2.csv> ...] \\\n",
        "         [--outdir=DIR] \\\n",
        "         [--kernels=k1,k2,...] \\\n",
        "         [--labels=l1,l2,...]\n",
        "\n",
        "Modes:\n",
        "  1 CSV   : single-snapshot (one bar per arch per kernel)\n",
        "  2 CSVs  : before/after (paired bars)\n",
        "  3-8 CSVs: N-way comparison\n",
        "\n",
        "Defaults:\n",
        "  --outdir  : current working directory\n",
        "  --kernels : intersection of kernels in all input CSVs\n",
        "  --labels  : basename(csv) with .csv stripped\n",
        sep = "", file = stderr()
    )
}

# ---- CLI parsing -------------------------------------------------

parse_args <- function(argv) {
    csvs <- character()
    flags <- list(outdir = NULL, kernels = NULL, labels = NULL)
    for (a in argv) {
        if (startsWith(a, "--")) {
            # Split on the FIRST '=' only. strsplit would split on
            # every '=', breaking legitimate values that contain '='
            # (e.g. --outdir=/tmp/run=v2/).
            eq <- regexpr("=", a, fixed = TRUE)
            if (eq < 4L) {
                # eq must be >=4: positions 1,2 are "--", position 3
                # is the name's first char, position 4+ is the '='.
                stop_with_usage(sprintf(
                    "malformed flag '%s' (expected --name=value)", a))
            }
            key <- substr(a, 3L, eq - 1L)
            val <- substr(a, eq + 1L, nchar(a))
            if (!nzchar(val)) {
                stop_with_usage(sprintf(
                    "malformed flag '%s' (expected --name=value)", a))
            }
            if (!key %in% names(flags)) {
                stop_with_usage(sprintf("unknown flag '--%s'", key))
            }
            flags[[key]] <- val
        } else {
            csvs <- c(csvs, a)
        }
    }
    if (length(csvs) < 1) {
        stop_with_usage("at least one CSV path required")
    }
    if (length(csvs) > MAX_RUNS) {
        stop_with_usage(sprintf(
            "too many input CSVs (%d); maximum is %d (palette cap)",
            length(csvs), MAX_RUNS))
    }
    list(csvs = csvs,
         outdir = flags$outdir %||% ".",
         kernels = flags$kernels,
         labels  = flags$labels)
}

# ---- CSV validation ---------------------------------------------

read_validated_csv <- function(path) {
    if (!file.exists(path)) {
        stop_loud(sprintf("input CSV does not exist: %s", path))
    }
    df <- tryCatch(
        utils::read.csv(path, header = TRUE, stringsAsFactors = FALSE,
                        colClasses = "character"),
        error = function(e) {
            stop_loud(sprintf(
                "could not read %s: %s", path, conditionMessage(e)))
        }
    )
    # Header check (exact match including order).
    if (!identical(colnames(df), EXPECTED_CSV_HEADER)) {
        stop_loud(sprintf(
            "CSV header mismatch in %s\n  expected: %s\n  got:      %s",
            path,
            paste(EXPECTED_CSV_HEADER, collapse = ","),
            paste(colnames(df), collapse = ",")
        ))
    }
    # trial: regex-check integer literal before coercion. as.integer()
    # on a character is fragile across R versions ("3.14" may NA-or-
    # truncate); regex is unambiguous.
    bad_trial <- which(!grepl("^-?[0-9]+$", df$trial))
    if (length(bad_trial) > 0) {
        line_no <- bad_trial[1] + 1L
        stop_loud(sprintf(
            "%s:%d: non-integer trial value '%s'",
            path, line_no, df$trial[bad_trial[1]]))
    }
    trial_num <- as.integer(df$trial)
    # time_ms: numeric coercion, then reject NA/Inf/NaN/<=0 explicitly.
    # Wall-times must be strictly positive: zero would survive as
    # log10(0) = -Inf and silently disappear from the plot; negative
    # values are nonsensical. volk_profile produces neither, but we
    # validate at the boundary the same way we reject "Inf".
    time_ms_num <- suppressWarnings(as.numeric(df$time_ms))
    bad_time <- which(is.na(time_ms_num) | !is.finite(time_ms_num) |
                      time_ms_num <= 0)
    if (length(bad_time) > 0) {
        line_no <- bad_time[1] + 1L
        stop_loud(sprintf(
            "%s:%d: non-positive or non-finite time_ms value '%s'",
            path, line_no, df$time_ms[bad_time[1]]))
    }
    df$trial   <- trial_num
    df$time_ms <- time_ms_num
    df
}

# ---- Aggregation -------------------------------------------------

summarise_runs <- function(dfs, labels) {
    stopifnot(length(dfs) == length(labels))
    parts <- mapply(
        function(df, lbl) {
            df |>
                dplyr::group_by(kernel, arch) |>
                dplyr::summarise(
                    median_ms = stats::median(time_ms),
                    mad_ms    = stats::mad(time_ms),
                    .groups = "drop"
                ) |>
                dplyr::mutate(run = factor(lbl, levels = labels))
        },
        dfs, labels, SIMPLIFY = FALSE
    )
    do.call(rbind, parts)
}

# ---- Plot rendering ---------------------------------------------

render_kernel_plot <- function(summary_df, kernel_name, outdir,
                                run_labels) {
    # Defensive filename validation: kernel_name becomes part of the
    # output path. Volk's actual kernel-naming convention is a strict
    # subset of [A-Za-z0-9_]+ and the longest real kernel is ~47
    # chars, so the cap below never trips on legitimate input — but
    # a hand-crafted CSV with "../etc/passwd" or a 5000-char name
    # would otherwise escape outdir or exceed NAME_MAX (255).
    if (!grepl("^[A-Za-z0-9_-]+$", kernel_name)) {
        stop_loud(sprintf(
            "kernel name '%s' contains characters unsafe for filename",
            kernel_name))
    }
    if (nchar(kernel_name) > 128L) {
        stop_loud(sprintf(
            "kernel name too long (%d chars, max 128): '%s'",
            nchar(kernel_name),
            substr(kernel_name, 1L, 60L)))
    }
    sub <- summary_df[summary_df$kernel == kernel_name, ]
    sub$arch <- factor(sub$arch,
                       levels = sort_archs_for_plot(unique(sub$arch)))
    n_runs <- length(run_labels)
    palette_used <- PALETTE_DARK2[seq_len(n_runs)]
    names(palette_used) <- run_labels

    # Median rendered as a horizontal line; MAD as whiskers around it.
    # Both use position_dodge so 2+ runs sit side-by-side per arch.
    # We deliberately do NOT draw bars: bar height to a y=0 baseline
    # forces log10 to clip at the lowest tick (1 ms) and visually
    # compresses the legitimate data range. With lines, the y-axis
    # auto-scales to the data — sub-percent MAD whiskers stay
    # invisibly tight, and the cluster structure is immediately
    # legible.
    dodge <- ggplot2::position_dodge(width = 0.6)
    p <- ggplot2::ggplot(sub,
            ggplot2::aes(x = arch, y = median_ms, color = run,
                         group = run)) +
        ggplot2::geom_errorbar(
            ggplot2::aes(ymin = pmax(median_ms - mad_ms, 1e-12),
                         ymax = median_ms + mad_ms),
            position = dodge, width = 0.25, linewidth = 0.6
        ) +
        # Horizontal "median line" via geom_errorbar with ymin == ymax;
        # the bracket collapses to two overlapping caps = one line.
        ggplot2::geom_errorbar(
            ggplot2::aes(ymin = median_ms, ymax = median_ms),
            position = dodge, width = 0.5, linewidth = 1.2
        ) +
        ggplot2::scale_y_log10() +
        ggplot2::scale_color_manual(values = palette_used) +
        ggplot2::labs(title = kernel_name,
                      x = "arch", y = "wall-time (ms, log scale)",
                      color = "run") +
        ggplot2::theme_bw(base_family = "sans")

    out_path <- file.path(outdir, sprintf("%s.png", kernel_name))
    ggplot2::ggsave(out_path, plot = p,
                    width = CANVAS_WIDTH_IN, height = CANVAS_HEIGHT_IN,
                    units = "in", dpi = CANVAS_DPI)
    # ggsave() emits warnings (not errors) on filename-too-long, mid-
    # run permission revocation, disk-full, etc. Verify the PNG was
    # actually written so the manifest cannot claim success when the
    # output is missing or zero-length.
    if (!file.exists(out_path) || file.size(out_path) == 0L) {
        stop_loud(sprintf(
            "ggsave failed to write '%s' (file missing or empty)",
            out_path))
    }
    invisible(out_path)
}

# ---- Manifest ----------------------------------------------------

write_manifest <- function(outdir, csv_paths, labels,
                            kernels_filter, kernels_count) {
    sha_of <- function(p) digest::digest(file = p, algo = "sha256")
    inputs_block <- paste(
        sprintf('  run%d (label "%s"): %s\n                                 (sha256: %s)',
                seq_along(csv_paths), labels, csv_paths,
                vapply(csv_paths, sha_of, character(1))),
        collapse = "\n"
    )
    script_path <- normalizePath(
        sub("^--file=", "", grep("^--file=",
            commandArgs(trailingOnly = FALSE), value = TRUE)[1]),
        mustWork = FALSE
    )
    # script_sha: system2(stdout=TRUE) returns character(0) on a
    # non-zero exit (no error), so a tryCatch wouldn't fire — inspect
    # the result directly. stderr=NULL discards git's "not a
    # repository" message.
    script_sha <- {
        raw <- suppressWarnings(system2(
            "git", c("-C", dirname(script_path),
                     "rev-parse", "HEAD"),
            stdout = TRUE, stderr = NULL))
        if (length(raw) == 0L) {
            "unknown (not in a git checkout)"
        } else {
            raw[1]
        }
    }
    # Mark dirty if the working tree's script file is untracked or
    # differs from HEAD. `git diff --quiet` would miss the untracked
    # case (untracked files do not appear in diff output), so use
    # `git status --porcelain`: empty output means committed-and-
    # clean; any other state produces a status line.
    dirty_flag <- tryCatch({
        porcelain <- suppressWarnings(system2(
            "git", c("-C", dirname(script_path),
                     "status", "--porcelain", "--",
                     script_path),
            stdout = TRUE, stderr = NULL))
        if (length(porcelain) > 0L) {
            " (+dirty)"
        } else {
            ""
        }
    }, error = function(e) "")
    kernels_str <- if (is.null(kernels_filter)) {
        "all (intersection)"
    } else {
        paste(kernels_filter, collapse = ", ")
    }
    text <- sprintf(
        paste0(
            "R version:        %s\n",
            "ggplot2 version:  %s\n",
            "dplyr version:    %s\n",
            "digest version:   %s\n",
            "script SHA:       %s%s\n",
            "inputs:\n%s\n",
            "kernels plotted:  %d; %s\n",
            "generated:        %s\n"
        ),
        R.version.string,
        as.character(utils::packageVersion("ggplot2")),
        as.character(utils::packageVersion("dplyr")),
        as.character(utils::packageVersion("digest")),
        script_sha, dirty_flag,
        inputs_block,
        kernels_count, kernels_str,
        format(Sys.time(), tz = "UTC", "%Y-%m-%dT%H:%M:%SZ")
    )
    # cat() rather than writeLines() — text already ends with '\n'
    # from the format string; writeLines would append another and
    # produce a ragged double newline at EOF.
    cat(text, file = file.path(outdir, "manifest.txt"))
}

# ---- Main --------------------------------------------------------

main <- function(argv) {
    if (length(argv) == 0) {
        print_usage()
        quit(status = 1, save = "no")
    }
    args <- parse_args(argv)

    # Resolve labels.
    labels <- if (!is.null(args$labels)) {
        ll <- trimws(strsplit(args$labels, ",", fixed = TRUE)[[1]])
        if (length(ll) != length(args$csvs)) {
            stop_loud(sprintf(
                "--labels has %d entries but %d CSV(s) provided",
                length(ll), length(args$csvs)))
        }
        if (any(!nzchar(ll))) {
            stop_loud("--labels contains an empty entry")
        }
        ll
    } else {
        sub("\\.csv$", "", basename(args$csvs))
    }
    if (anyDuplicated(labels)) {
        stop_loud(sprintf(
            "labels must be unique; got: %s",
            paste(labels, collapse = ", ")))
    }

    # Validate and read every CSV.
    dfs <- lapply(args$csvs, read_validated_csv)

    # Aggregate.
    summary_df <- summarise_runs(dfs, labels)

    # Resolve kernel set. kernels_filter stays NULL in no-filter
    # mode so the manifest can record "all (intersection)" rather
    # than enumerating every kernel name.
    kernels_filter <- NULL
    if (!is.null(args$kernels)) {
        wanted <- trimws(strsplit(args$kernels, ",", fixed = TRUE)[[1]])
        if (any(!nzchar(wanted))) {
            stop_loud("--kernels contains an empty entry")
        }
        # Dedupe so a typo like --kernels=k1,k2,k1 doesn't render
        # the same PNG twice and inflate the manifest count.
        wanted <- unique(wanted)
        # Each wanted kernel must appear in *every* input CSV.
        for (i in seq_along(dfs)) {
            absent <- setdiff(wanted, unique(dfs[[i]]$kernel))
            if (length(absent) > 0) {
                stop_loud(sprintf(
                    "kernel '%s' from --kernels filter is absent from %s",
                    absent[1], args$csvs[i]))
            }
        }
        kernels_to_plot <- wanted
        kernels_filter  <- wanted
    } else {
        kernels_to_plot <- Reduce(intersect,
                                  lapply(dfs, function(d) unique(d$kernel)))
        if (length(kernels_to_plot) == 0) {
            stop_loud("no kernels in common across all input CSVs")
        }
    }

    # Output dir.
    if (!dir.exists(args$outdir)) {
        ok <- dir.create(args$outdir, recursive = TRUE,
                         showWarnings = FALSE)
        if (!ok) {
            stop_loud(sprintf(
                "could not create --outdir '%s'", args$outdir))
        }
    }
    if (file.access(args$outdir, mode = 2) != 0) {
        stop_loud(sprintf(
            "--outdir '%s' is not writable", args$outdir))
    }

    # Render.
    for (k in sort(kernels_to_plot)) {
        render_kernel_plot(summary_df, k, args$outdir, labels)
    }
    write_manifest(args$outdir, args$csvs, labels,
                   kernels_filter, length(kernels_to_plot))

    cat(sprintf("wrote %d PNG(s) and manifest.txt to %s\n",
                length(kernels_to_plot), args$outdir))
}

if (!interactive()) {
    main(commandArgs(trailingOnly = TRUE))
}
