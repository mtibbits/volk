.function volk_32fc_deinterleave_32f_x2_a_orc_impl
.source 8 src
.dest 4 dst_i float
.dest 4 dst_q float
splitql dst_q, dst_i, src
