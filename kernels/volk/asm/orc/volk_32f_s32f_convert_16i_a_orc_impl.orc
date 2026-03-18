.function volk_32f_s32f_convert_16i_a_orc_impl
.source 4 src float
.dest 2 dst int16_t
.floatparam 4 scalar
.temp 4 ftmp
.temp 4 ltmp
mulf ftmp, src, scalar
convfl ltmp, ftmp
convssslw dst, ltmp
