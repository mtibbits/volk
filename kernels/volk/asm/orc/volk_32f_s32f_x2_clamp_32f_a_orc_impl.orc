.function volk_32f_s32f_x2_clamp_32f_a_orc_impl
.source 4 src float
.dest 4 dst float
.floatparam 4 min_val
.floatparam 4 max_val
.temp 4 tmp
maxf tmp, src, min_val
minf dst, tmp, max_val
