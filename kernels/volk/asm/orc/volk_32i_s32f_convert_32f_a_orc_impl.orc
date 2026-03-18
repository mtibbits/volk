.function volk_32i_s32f_convert_32f_a_orc_impl
.source 4 src int32_t
.dest 4 dst float
.floatparam 4 invscalar
.temp 4 flsrc
convlf flsrc, src
mulf dst, flsrc, invscalar
