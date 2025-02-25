CPP=CC
CFLAGS=-lm
OPTFLAGS=-O3 -ffast-math -ftree-vectorize -mavx2 -flto
DEBUGFLAGS=-g -pg
PARFLAGS= -fopenmp

NVCC=nvcc
NVCCFLAGS=-DCUDA -O3 -Xptxas="-dlcm=ca" -fmad=true

PYTHON=python3

all: h_gpu_sep gpu_multikernel h_gpu gpu serial separable_serial block_serial basic_serial sobel_serial

h_gpu_sep: build/h_gpu_sep
gpu_multikernel: build/gpu_multikernel
h_gpu: build/h_gpu
gpu: build/gpu
separable_serial: build/separable_serial
separable_parallel: build/separable_parallel
block_serial: build/block_serial
serial: build/serial
basic_serial: build/basic_serial
sobel_serial: build/sobel_serial

build/h_gpu_sep: common/main.cpp utils/utils.cpp gpu/h_gpu_sep.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

build/gpu_multikernel: common/main.cpp utils/utils.cpp gpu/gpu_multikernel.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

build/h_gpu: common/main.cpp utils/utils.cpp gpu/h_gpu.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

build/gpu: common/main.cpp utils/utils.cpp gpu/gpu.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

build/separable_parallel: common/main.cpp utils/utils.cpp omp/separable_parallel.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS) $(OPTFLAGS) $(PARFLAGS) -DPARALLEL

build/separable_serial: common/main.cpp utils/utils.cpp serial/separable_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS) $(OPTFLAGS)

build/block_serial: common/main.cpp utils/utils.cpp serial/block_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS) $(OPTFLAGS)

build/serial: common/main.cpp utils/utils.cpp serial/serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS) $(OPTFLAGS)

build/basic_serial: common/main.cpp utils/utils.cpp serial/basic_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS) 

build/sobel_serial: common/main.cpp utils/utils.cpp serial/sobel_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS) 

.PHONY: clean

clean:
	rm -f build/*.out
	rm -f build/*.o
	rm -f build/*.jpg
	rm -f build/*.txt