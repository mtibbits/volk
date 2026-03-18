.function volk_32f_64f_multiply_64f_a_orc_impl
.source 4 src1 float
.source 8 src2 double
.dest 8 dst double
.temp 8 tmp
convfd tmp, src1
muld dst, tmp, src2
