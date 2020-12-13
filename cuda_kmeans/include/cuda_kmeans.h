#ifndef CUDA_KMEANS
#define CUDA_KMEANS

#include <cuda_runtime.h>
#include "utils.h"

class cuda_kmeans{
    private:
        int k, n, d;

        void init_centroids();
        void assign_to_centroids(int* assign, performance_t& p);
        void update_centroids(int* assign, performance_t& p);
            
    public:
        point_t data, centroids;
        void execute(int iterations, performance_t& p);

        /**
         * This constructor initialises the parameters
         * for kmeans.
         */
        cuda_kmeans(point_t data, point_t centroids,int k, int n, int d){
            this->k = k;
            this->n = n;
            this->d = d;

            this->data = data;
            this->centroids = centroids;
        }
};


#endif