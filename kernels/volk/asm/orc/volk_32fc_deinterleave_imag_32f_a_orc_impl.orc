.function volk_32fc_deinterleave_imag_32f_a_orc_impl
.source 8 src
.dest 4 dst float
.temp 4 real
splitql dst, real, src
