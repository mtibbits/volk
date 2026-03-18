.function volk_16i_convert_8i_a_orc_impl
.source 2 src int16_t
.dest 1 dst int8_t
.temp 2 tmp
shrsw tmp, src, 8
convssswb dst, tmp
