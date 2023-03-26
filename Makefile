CC=nvcc

all: 001

001: 001_simple.cu
	$(CC) -o simple.exe 001_simple.cu

002: 002_sharedmem.cu
	$(CC) -o sharedmem.exe 002_sharedmem.cu

003: 003_matmul.cu
	$(CC) -o matmul.exe 003_matmul.cu

clean:
	rm *exe