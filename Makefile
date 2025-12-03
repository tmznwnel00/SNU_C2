
shared:
	nvcc -c shared_kernels.cu -o shared_kernels.o
c1:
	nvcc -c -O3 coclustering_1.cu -o coc1.o -lcusparse -lcublas

c2:
	nvcc -c -O3 coclustering_2.cu -o coc2.o -lcusparse -lcublas -lcusolver -lcurand

c3:
	nvcc -c -O3 coclustering_3.cu -o coc3.o -lcublas -lcurand

c4:
	nvcc -c -O3 coclustering_4.cu -o coc4.o -lcublas

all:
	nvcc -c shared_kernels.cu -o shared_kernels.o
	nvcc -c -O3 coclustering_4.cu -o coc4.o -lcublas 
	nvcc -c -O3 coclustering_2.cu -o coc2.o -lcusparse -lcublas -lcusolver -lcurand 
	nvcc -c -O3 coclustering_3.cu -o coc3.o -lcublas -lcurand 
	nvcc -c -O3 coclustering_4.cu -o coc4.o -lcublas

main:
	nvcc cocluster_main.cu coc1.o coc2.o coc3.o coc4.o shared_kernels.o \
    -o cocluster \
    -lcusparse -lcublas -lcusolver -lcurand
clean:
	rm -f *.o
