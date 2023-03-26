CC=nvcc

all: 001

001: 001_simple.cu
	$(CC) -o simple 001_simple.cu