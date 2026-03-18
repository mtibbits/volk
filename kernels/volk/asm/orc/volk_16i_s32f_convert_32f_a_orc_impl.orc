.function volk_16i_s32f_convert_32f_a_orc_impl
.source 2 src int16_t
.dest 4 dst float
.floatparam 4 scalar
.temp 2 ssrc
.temp 4 lsrc
.temp 4 flsrc
convswl lsrc, src
convlf flsrc, lsrc
mulf dst, flsrc, scalar
