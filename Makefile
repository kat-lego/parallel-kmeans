CFLAGS = -std=c++11
CC = g++
NVCC = nvcc
MPICC = mpicxx

serial: serial_kmeans.cpp
	$(CC) $(CFLAGS) serial_kmeans.cpp -o skmeans

cuda: cuda_kmeans.cu
	$(NVCC) $(CFLAGS) cuda_kmeans.cu -o ckmeans

mpi: mpi_kmeans.cpp
	$(MPICC) $(CFLAGS) mpi_kmeans.cpp -o mkmeans

clean:
	rm skmeans ckmeans mkmeans


