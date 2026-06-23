/* -*- c++ -*- */
/*
 * Copyright 2018-2020 Free Software Foundation, Inc.
 *
 * This file is part of VOLK
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


#include "volk_option_helpers.h"

#include <limits.h>  // IWYU pragma: keep
#include <cerrno>    // IWYU pragma: keep
#include <cmath>     // for std::isfinite
#include <cstdlib>   // IWYU pragma: keep
#include <cstring>   // IWYU pragma: keep
#include <exception> // for exception
#include <iostream>  // for operator<<, endl, basic_ostream, cout, ostream
#include <utility>   // for pair

/*
 * Option type
 */
option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   void (*t_callback)())
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      callback(t_callback)
{
    option_type = VOID_CALLBACK;
}

option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   void (*t_callback)(int))
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      callback((void (*)())t_callback)
{
    option_type = INT_CALLBACK;
}

option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   void (*t_callback)(float))
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      callback((void (*)())t_callback)
{
    option_type = FLOAT_CALLBACK;
}

option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   void (*t_callback)(bool))
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      callback((void (*)())t_callback)
{
    option_type = BOOL_CALLBACK;
}

option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   void (*t_callback)(std::string))
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      callback((void (*)())t_callback)
{
    option_type = STRING_CALLBACK;
}

option_t::option_t(std::string t_longform,
                   std::string t_shortform,
                   std::string t_msg,
                   std::string t_printval)
    : longform("--" + t_longform),
      shortform("-" + t_shortform),
      msg(t_msg),
      printval(t_printval)
{
    option_type = STRING;
}


/*
 * Option List
 */

option_list::option_list(std::string program_name) : d_program_name(program_name)
{
    d_internal_list = std::vector<option_t>();
}


void option_list::add(option_t opt) { d_internal_list.push_back(opt); }

void option_list::parse(int argc, char** argv)
{
    for (int arg_number = 0; arg_number < argc; ++arg_number) {
        bool matched = false;
        for (std::vector<option_t>::iterator this_option = d_internal_list.begin();
             this_option != d_internal_list.end();
             this_option++) {
            int int_val = INT_MIN;
            if (this_option->longform == std::string(argv[arg_number]) ||
                this_option->shortform == std::string(argv[arg_number])) {
                matched = true;

                if (d_present_options.count(this_option->longform) == 0) {
                    d_present_options.insert(
                        std::pair<std::string, int>(this_option->longform, 1));
                } else {
                    d_present_options[this_option->longform] += 1;
                }
                switch (this_option->option_type) {
                case VOID_CALLBACK:
                    this_option->callback();
                    break;
                case INT_CALLBACK:
                    try {
                        if (arg_number + 1 >= argc) {
                            std::cerr << "Error: option '" << argv[arg_number]
                                      << "' requires a value" << std::endl;
                            exit(1);
                        }
                        {
                            char* endptr = nullptr;
                            errno = 0;
                            long long_val = strtol(argv[++arg_number], &endptr, 10);
                            if (endptr == argv[arg_number] || *endptr != '\0' ||
                                errno == ERANGE || long_val < INT_MIN ||
                                long_val > INT_MAX) {
                                std::cerr << "Error: option '" << argv[arg_number - 1]
                                          << "' requires a numeric value, got '"
                                          << argv[arg_number] << "'" << std::endl;
                                exit(1);
                            }
                            int_val = (int)long_val;
                            ((void (*)(int))this_option->callback)(int_val);
                        }
                    } catch (std::exception& exc) {
                        std::cerr << "An int option can only receive a number"
                                  << std::endl;
                        throw std::exception();
                    };
                    break;
                case FLOAT_CALLBACK:
                    try {
                        if (arg_number + 1 >= argc) {
                            std::cerr << "Error: option '" << argv[arg_number]
                                      << "' requires a value" << std::endl;
                            exit(1);
                        }
                        {
                            char* endptr = nullptr;
                            errno = 0;
                            double double_val = strtod(argv[++arg_number], &endptr);
                            float float_val = (float)double_val;
                            if (endptr == argv[arg_number] || *endptr != '\0' ||
                                errno == ERANGE || !std::isfinite(float_val)) {
                                std::cerr << "Error: option '" << argv[arg_number - 1]
                                          << "' requires a numeric value, got '"
                                          << argv[arg_number] << "'" << std::endl;
                                exit(1);
                            }
                            ((void (*)(float))this_option->callback)(float_val);
                        }
                    } catch (std::exception& exc) {
                        std::cerr << "A float option can only receive a number"
                                  << std::endl;
                        throw std::exception();
                    };
                    break;
                case BOOL_CALLBACK:
                    if (arg_number == (argc - 1)) {
                        int_val = 1;
                    } else {
                        const char* next_arg = argv[arg_number + 1];
                        if (strncmp(next_arg, "-", 1) == 0) {
                            int_val = 1;
                        } else if (strncmp(next_arg, "true", 4) == 0) {
                            int_val = 1;
                            ++arg_number;
                        } else if (strncmp(next_arg, "false", 5) == 0) {
                            int_val = 0;
                            ++arg_number;
                        } else if (next_arg[0] >= '0' && next_arg[0] <= '9') {
                            int_val = (bool)atoi(argv[++arg_number]);
                        } else {
                            int_val = 1;
                        }
                    }
                    if (int_val) {
                        ((void (*)(bool))this_option->callback)(int_val);
                    }
                    break;
                case STRING_CALLBACK:
                    try {
                        if (arg_number + 1 >= argc) {
                            std::cerr << "Error: option '" << argv[arg_number]
                                      << "' requires a value" << std::endl;
                            exit(1);
                        }
                        ((void (*)(std::string))this_option->callback)(
                            argv[++arg_number]);
                    } catch (std::exception& exc) {
                        throw std::exception();
                    };
                    break;
                case STRING:
                    std::cout << this_option->printval << std::endl;
                    break;
                }
            }
        }
        if (std::string("--help") == std::string(argv[arg_number]) ||
            std::string("-h") == std::string(argv[arg_number])) {
            matched = true;
            d_present_options.insert(std::pair<std::string, int>("--help", 1));
            help();
        }
        if (!matched && arg_number > 0 && argv[arg_number][0] == '-') {
            std::cerr << "Warning: unrecognized option '" << argv[arg_number] << "'"
                      << std::endl;
        }
    }
}

bool option_list::present(std::string option_name)
{
    if (d_present_options.count("--" + option_name)) {
        return true;
    } else {
        return false;
    }
}

void option_list::help()
{
    std::cout << d_program_name << std::endl;
    std::cout << "  -h [ --help ] \t\tdisplay this help message" << std::endl;
    for (std::vector<option_t>::iterator this_option = d_internal_list.begin();
         this_option != d_internal_list.end();
         this_option++) {
        std::string help_line("  ");
        if (this_option->shortform == "-") {
            help_line += this_option->longform + " ";
        } else {
            help_line += this_option->shortform + " [ " + this_option->longform + " ]";
        }

        while (help_line.size() < 32) {
            help_line += " ";
        }
        help_line += this_option->msg;
        std::cout << help_line << std::endl;
    }
}
