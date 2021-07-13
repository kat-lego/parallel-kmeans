CFLAGS = -std=c++11
CC = g++
NVCC = nvcc
MPICC = mpicxx

all: serial cuda mpi

serial: src/serial_kmeans.cpp
	$(CC) $(CFLAGS) src/serial_kmeans.cpp -o skmeans.o

cuda: src/cuda_kmeans.cu
	$(NVCC) $(CFLAGS) src/cuda_kmeans.cu -o ckmeans.o

mpi: src/mpi_kmeans.cpp
	$(MPICC) $(CFLAGS) src/mpi_kmeans.cpp -o mkmeans.o

clean:
	rm *.o


