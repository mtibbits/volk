.function volk_32fc_conjugate_32fc_a_orc_impl
.source 8 src
.dest 8 dst
.floatparam 4 negone
.temp 4 real
.temp 4 imag
splitql imag, real, src
mulf imag, imag, negone
mergelq dst, real, imag
