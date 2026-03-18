.function volk_32f_s32f_convert_32i_a_orc_impl
.source 4 src float
.dest 4 dst int32_t
.floatparam 4 scalar
.temp 4 tmp
mulf tmp, src, scalar
convfl dst, tmp
