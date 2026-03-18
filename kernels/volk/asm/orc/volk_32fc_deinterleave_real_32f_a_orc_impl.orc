.function volk_32fc_deinterleave_real_32f_a_orc_impl
.source 8 src
.dest 4 dst float
.temp 4 imag
splitql imag, dst, src
