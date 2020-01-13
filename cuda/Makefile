CUDA_FLAGS = -arch=sm_30 -O3 -save-temps

all: collatz

collatz: collatz.cu
	nvcc ${CUDA_FLAGS} collatz.cu -o $@

clean:
	rm -f collatz *.i *.ii *.cudafe* *.cubin* *.fatbin* *.hash *.module_id *_dlink.* *.ptx *.o
