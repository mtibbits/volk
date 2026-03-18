.function volk_32f_s32f_convert_8i_a_orc_impl
.source 4 src float
.dest 1 dst int8_t
.floatparam 4 scalar
.temp 4 ftmp
.temp 4 ltmp
.temp 2 wtmp
mulf ftmp, src, scalar
convfl ltmp, ftmp
convssslw wtmp, ltmp
convssswb dst, wtmp
