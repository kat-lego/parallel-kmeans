#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include "../include/utils.h"
#include "../include/mpi_kmeans.h"

void mpi_kmeans::execute(int iterations, performance_t& p){
    double totaltime = MPI_Wtime(); 


    int* assign = new int[n];
    int it = 0;

    //create a type for a data point
    MPI_Datatype point_type;
    MPI_Type_contiguous(d, MPI_FLOAT, &point_type);
    MPI_Type_commit(&point_type);

    
    // initial centroids
    if(id==0){
        init_centroids();
        // print_points(centroids, k, d);
    }

    MPI_Bcast(centroids, k, point_type, 0, MPI_COMM_WORLD);
    double t1, t2; 

    while(it<iterations){

        t1 = MPI_Wtime(); 
        assign_to_centroids(assign);
        
        // broadcast assign[] to each task
        int s=n/num_p;
        for(int i=0;i<num_p;i++){
            MPI_Bcast(assign+s*i, s, MPI_INT, i, MPI_COMM_WORLD);
        }
        t2 = MPI_Wtime();
        p.runtime1+=(t2-t1)*1000; 
        
        t1 = MPI_Wtime();
        update_centroids(assign);
        
        // broadcast centroids[] to each task
        s = (k>num_p)?k/num_p:1;
        int up = (s==1)?k:num_p;
        for(int i=0;i<up;i++){
            MPI_Bcast(centroids+s*i*d, s, point_type, i, MPI_COMM_WORLD);
        }
        t2 = MPI_Wtime();
        p.runtime2+=(t2-t1)*1000;

        it++;
    }

    // print_points(centroids, k, d);

    p.runtime3+= (MPI_Wtime()-totaltime)*1000;
    
    MPI_Type_free(&point_type);
    delete[] assign;
}

void mpi_kmeans::init_centroids(){
    for(int i=0;i<k*d;i++)centroids[i] = data[i];
}

void mpi_kmeans::assign_to_centroids(int* assign){
    float sum = 0;
    float min;
    int min_c;

    int s=n/num_p;
    int up = (s+s*id<n)?s+s*id:n;
    for(int i=s*id; i<up; i++){ 
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

void mpi_kmeans::update_centroids(int* assign){
    int count;
    
    int s = (k>num_p)?k/num_p:1;

    int up = (s*id+s<k)?s*id+s:k;
    for(int i=0;i<k;i++)
        for(int l=0;l<d;l++)
            centroids[i*d+l] = 0; //clear the array

    for(int i=s*id;i<up;i++){
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
