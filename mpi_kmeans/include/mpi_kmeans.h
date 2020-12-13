#ifndef MPI_KMEANS
#define MPI_MEANS

#include <string>
#include <mpi.h>
#include "utils.h"

class mpi_kmeans{
    private:
        int num_p, id;
        int k, n, d;

        void init_centroids();
        void assign_to_centroids(int* assign);
        void update_centroids(int* assign);
    
    public:
        point_t data, centroids;
        void execute(int iterations, performance_t& p);

        /**
         * This constructor initialises the parameters
         * for kmeans.
         */
        mpi_kmeans(point_t data, point_t centroids,int k, int n, int d){
            this->k = k;
            this->n = n;
            this->d = d;

            this->data = data;
            this->centroids = centroids;

            MPI_Comm_size(MPI_COMM_WORLD, &num_p);
            MPI_Comm_rank(MPI_COMM_WORLD, &id);

        }

};


#endif