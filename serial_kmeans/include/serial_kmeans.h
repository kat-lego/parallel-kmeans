#ifndef SERIAL_KMEANS
#define SERIAL_KMEANS

#include <string>
#include "utils.h"

class serial_kmeans{
    private:
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
        serial_kmeans(point_t data, point_t centroids,int k, int n, int d){
            this->k = k;
            this->n = n;
            this->d = d;

            this->data = data;
            this->centroids = centroids;
        }
};


#endif