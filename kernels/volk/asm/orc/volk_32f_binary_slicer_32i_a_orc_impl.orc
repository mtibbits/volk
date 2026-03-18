.function volk_32f_binary_slicer_32i_a_orc_impl
.source 4 src
.dest 4 dst
.temp 4 tmp
.const 4 one 1
shrul tmp, src, 31
xorl dst, tmp, one
