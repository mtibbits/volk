.function volk_32fc_magnitude_squared_32f_a_orc_impl
.source 8 src
.dest 4 dst float
.temp 4 real
.temp 4 imag
splitql imag, real, src
mulf real, real, real
mulf imag, imag, imag
addf dst, real, imag
