#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>  

#include "../include/utils.h"


void print_points(point_t points, int n, int d){
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    printf("=====================================================================================================\n");
    for(int i=0; i<n; i++){
        printf("[%d:",id );
        for(int j=0; j<d; j++){
            printf("%0.3f, ", points[i*d+j]);
        }
        printf("]\n");
    }
    printf("=====================================================================================================\n\n");

}

void init_test_points(point_t points, int n, int d){
    srand(0);
    for(int i=0;i<n*d;i++){
        points[i] = -50+rand()%(100);
    }
}

void performance_t::clear(){
    runtime1 = 0;
    runtime2 = 0;
    runtime3 = 0;
}

