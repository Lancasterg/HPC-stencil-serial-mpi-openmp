stencil: stencil.c
	mpiicc -cc=icc -std=c99 -O3 -D NOALIAS -xAVX -restrict -Wall $^ -o $@

