#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#include "../include/utils.h"
#include "../include/serial_kmeans.h"


void serial_kmeans::execute(int iterations, performance_t& p){
    cudaTime_t t, t2;
    t2.start_time();
    
    int* assign = new int[n];
    int it = 0;
    float r1, r2, r3;

    // initial centroids
    init_centroids();

    while(it<iterations){
        t.start_time();
        assign_to_centroids(assign);
        t.stop_time(r1);
        p.runtime1+=r1;

        t.start_time();
        update_centroids(assign);
        t.stop_time(r2);
        p.runtime2+=r2;
        it++;
    }

    // print_points(centroids, k, 16);
    t2.stop_time(r3);
    p.runtime3+=r3;
    delete[] assign;
}

void serial_kmeans::init_centroids(){
    for(int i=0;i<k*d;i++)centroids[i] = data[i];
    // init_test_points(centroids, k, d);
}

void serial_kmeans::assign_to_centroids(int* assign){
    float sum = 0;
    float min;
    int min_c;

    for(int i=0; i<n; i++){
        min = MAX_DISTANCE;
        for( int j=0; j<k;j++){
            sum = 0;
            for(int l=0; l<d; l++){
                sum+= (data[i*d+l]-centroids[j*d+l])*(data[i*d+l]-centroids[j*d+l]);
            }

            sum = sqrt(sum);
            if(sum<=min){
                min = sum;
                min_c = j;   
            }
        }
        assign[i] = min_c;
    }
}

void serial_kmeans::update_centroids(int* assign){
    int count;
    for(int l=0;l<d*k;l++)centroids[l] = 0; //clear the array

    for(int i=0;i<k;i++){
        count = 0;
        for(int j=0;j<n;j++){
            if(assign[j]==i){
                count++;
                for(int l=0;l<d;l++){
                    centroids[i*d+l] += data[j*d+l];
                }
            }
        }

        for(int l=0;l<d;l++)centroids[i*d+l]/=count;
    }
}
