#include <string>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>  
#include <time.h>

#include "../include/utils.h"


void print_points(point_t points, int n, int d){
    printf("=====================================================================================================\n");
    for(int i=0; i<n; i++){
        printf("[");
        for(int j=0; j<d; j++){
            printf("%0.3f, ", points[i*d+j]);
        }
        printf("]\n");
    }
    printf("=====================================================================================================\n\n");

}

void init_test_points(point_t points, int n, int d){
    srand(time(0));
    for(int i=0;i<n*d;i++){
        points[i] = -50+rand()%(100);
    }
}


void cudaTime_t::start_time(){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
}


void cudaTime_t::stop_time(float& time){
    cudaEventRecord(stop,0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );
}

void performance_t::clear(){
    runtime1 = 0;
    runtime2 = 0;
    runtime3 = 0;
}
