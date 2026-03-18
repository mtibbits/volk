.function volk_32fc_32f_add_32fc_a_orc_impl
.source 8 src1
.source 4 src2 float
.dest 8 dst
.temp 4 real
.temp 4 imag
splitql imag, real, src1
addf real, real, src2
mergelq dst, real, imag
