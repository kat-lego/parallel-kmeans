#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#include "../include/utils.h"
#include "../include/cuda_kmeans.h"

#define BLOCKWIDTH 512
#define STATIC_GSIZE 8
#define MAXUNITS 268435455

__global__ void cuda_compute_distances(point_t sums, point_t data, point_t centroids, int k, int d);
__global__ void cuda_compute_argmins(int* argmin, point_t sums, int k);

__global__ void cuda_clear(point_t centroids, int size);
__global__ void cuda_centroid_sums(point_t centroids, int* counts, point_t data, int* assign, int k, int n, int d);
__global__ void cuda_centroid_means(point_t centroids, int* counts, int k, int d);


void cuda_kmeans::execute(int iterations,  performance_t& p){
    cudaTime_t t;
    t.start_time();

    int* assign = new int[n];
    int it = 0;
    float r;

    // initial centroids
    init_centroids();

    while(it<iterations){
        assign_to_centroids(assign, p);
        update_centroids(assign, p);
        it++;
    }

    //final centroids
    // print_points(centroids, k, d);

    t.stop_time(r);
    p.runtime3+=r;
    
    delete[] assign;
    
}

void cuda_kmeans::init_centroids(){
    for(int i=0;i<k*d;i++)centroids[i] = data[i];
    // init_test_points(centroids, k, d);
}

void cuda_kmeans::assign_to_centroids(int* assign, performance_t& p){
    //declare pointers for device memory
    point_t dev_data, dev_centroids, dev_distances;
    int* dev_assign;


    //allocate device memory
    size_t size_data = n*d*sizeof(element_t);
    size_t size_centroids = k*d*sizeof(element_t);
    size_t size_assign = n*sizeof(int);
    size_t size_distances = n*k*sizeof(element_t);
    cudaMalloc( (void**)&dev_data, size_data  );
    cudaMalloc( (void**)&dev_centroids, size_centroids );
    cudaMalloc( (void**)&dev_assign,  size_assign );
    cudaMalloc( (void**)&dev_distances,  size_distances );


    //copy over the device memory
    cudaMemcpy( dev_data, data, size_data , cudaMemcpyHostToDevice);
    cudaMemcpy( dev_centroids, centroids, size_centroids , cudaMemcpyHostToDevice);

    // Kernel parameters
    //   - number of threads in a block is given by BLOCKWITDH
    //   - number of blocks is given by n*k
    int nblocks = n*k;

    if(MAXUNITS<nblocks){
        printf("Error in ckmeans::assign_to_cluster\n");
    }

    cudaTime_t t;
    float time;
    t.start_time();

    //Run the kernels
    cuda_compute_distances<<< nblocks,BLOCKWIDTH>>>(dev_distances, dev_data, dev_centroids, k, d);
    cuda_compute_argmins<<<n, BLOCKWIDTH>>>(dev_assign, dev_distances, k);

    t.stop_time(time);
    p.runtime1+=time;
    // copy to host
    cudaMemcpy( assign, dev_assign, size_assign, cudaMemcpyDeviceToHost);

    //free memory
    cudaFree(dev_assign);
    cudaFree(dev_centroids);
    cudaFree(dev_data);
    cudaFree(dev_distances);

}


void cuda_kmeans::update_centroids(int* assign, performance_t& p){
    //declare pointers for device memory
    point_t dev_data, dev_centroids;
    int* dev_assign, *dev_counts;

    //allocate device memory
    size_t size_data = n*d*sizeof(element_t);
    size_t size_centroids = k*d*sizeof(element_t);
    size_t size_assign = n*sizeof(int);
    size_t size_counts = k*sizeof(int);
    cudaMalloc( (void**)&dev_data, size_data  );
    cudaMalloc( (void**)&dev_centroids, size_centroids );
    cudaMalloc( (void**)&dev_assign,  size_assign );
    cudaMalloc( (void**)&dev_counts,  size_counts );
    
    //copy over the device memory
    cudaMemcpy( dev_data, data, size_data , cudaMemcpyHostToDevice);
    cudaMemcpy( dev_assign, assign, size_assign , cudaMemcpyHostToDevice);

    cudaTime_t t;
    float time;
    t.start_time();

    //centroid updates
    int nblocks = (k*d+BLOCKWIDTH-1)/BLOCKWIDTH;
    int nblocks2 = k*STATIC_GSIZE;
    cuda_clear<<<nblocks,BLOCKWIDTH>>>(dev_centroids, k*d);
    cuda_centroid_sums<<<nblocks2,BLOCKWIDTH, (d+1)*sizeof(element_t)>>>(dev_centroids, dev_counts, dev_data, dev_assign, k, n, d);
    cuda_centroid_means<<<k,BLOCKWIDTH>>>(dev_centroids, dev_counts, k, d);

    t.stop_time(time);
    p.runtime2+=time;

    // copy to host
    cudaMemcpy( centroids, dev_centroids, size_centroids, cudaMemcpyDeviceToHost);

    // //free memory
    cudaFree(dev_assign);
    cudaFree(dev_centroids);
    cudaFree(dev_data);
    cudaFree(dev_counts);
}

/**
* Compute the distances between each data point and centroids. 
*/
__global__ void cuda_compute_distances(point_t sums, point_t data, point_t centroids, int k, int d){
    
    //get the group id, block id and thread id.
    int tid = threadIdx.x; int bid = blockIdx.x%k;
    int gid = blockIdx.x/k;
    
    //shared memory array to store the terms from the euclidean distance
    __shared__ element_t ds[BLOCKWIDTH];
    ds[tid] = 0;

    //just some peace of mind
    if(tid>=d)return;

    //putting together the diffs
    while(tid<d){
        ds[tid%BLOCKWIDTH]+= (data[gid*d+tid]-centroids[bid*d+tid])*(data[gid*d+tid]-centroids[bid*d+tid]);
        tid+=BLOCKWIDTH;
    }
    tid = threadIdx.x; //reset tid
    __syncthreads();

    //putting together the sums (reduction)
    int slab = (d<BLOCKWIDTH)?d>>1:BLOCKWIDTH>>1;
    while(slab>=1){
        if(tid<slab)
            ds[tid] += ds[tid+slab];
        
        slab = slab>>1;
        __syncthreads();
    }

    // if(gid==1  && tid==0){
    //     printf("ds[%d] = %0.3f\n", bid, ds[0]);
    // }
    sums[gid*k+bid] = sqrt(ds[0]);

}

/**
* Compute the argmins for the distances
*/
__global__ void cuda_compute_argmins(int* argmin, point_t sums, int k){
    int tid = threadIdx.x; int bid = blockIdx.x;

    __shared__ element_t dmins[BLOCKWIDTH];
    __shared__ int dargs[BLOCKWIDTH];
    
    // initialise to max values
    if(tid>=k)return;
    while(tid<k){
        dmins[tid]= MAX_DISTANCE;
        tid+=BLOCKWIDTH;
    }
    __syncthreads();

    // serial min 
    tid = threadIdx.x;
    while(tid<k){
        // if(bid==2){
        //     printf("sum for centroid %d is %0.2f\n", tid, sums[bid*k+tid]);
        // }

        if(dmins[tid%BLOCKWIDTH]>sums[bid*k+tid]){
            dmins[tid%BLOCKWIDTH] = sums[bid*k+tid];
            dargs[tid%BLOCKWIDTH] = tid;
        }

        tid+=BLOCKWIDTH;
    }
    __syncthreads();

    //putting together the sums (reduction)
    tid = threadIdx.x;
    int slab = (k<BLOCKWIDTH)?k:BLOCKWIDTH;
    slab = slab>>1;
    while(slab>=1){
        if(tid<slab && dmins[tid]>dmins[tid+slab]){
            dmins[tid] = dmins[tid+slab];
            dargs[tid] = dargs[tid+slab];
        }
        
        slab = slab>>1;
        __syncthreads();
    }

    if(tid==0){
        // printf("min %d, %0.2f, %d\n", bid, dmins[tid], dargs[tid]);
        argmin[bid] = dargs[tid];
    }
 
}



__global__ void cuda_clear(point_t centroids, int size){
    int id = blockIdx.x*blockDim.x+ threadIdx.x;
    if(id>=size)return;
    centroids[id]=0;
}

__global__ void cuda_centroid_sums(point_t centroids, int* counts, point_t data, int* assign, int k, int n, int d){
    int tid = threadIdx.x; int bid = blockIdx.x%STATIC_GSIZE;
    int gid = blockIdx.x/STATIC_GSIZE;
    
    extern __shared__ element_t s[];
    
    if(tid>=d)return;
    
    //reset the stuff in s to 0
    while(tid<d){
        s[tid] =0;
        tid+=BLOCKWIDTH;
    }
    tid = threadIdx.x;
    __syncthreads();

    s[d] =0; //s[d] is the counter

    while(bid<n){
        if(assign[bid]==gid){
            s[d]++;
            while(tid<d){
                s[tid]+=data[bid*d+tid];
                tid+=BLOCKWIDTH;
            }
            __syncthreads();
        }

        tid = threadIdx.x;
        bid+=STATIC_GSIZE;
    }
    tid = threadIdx.x;
    bid=blockIdx.x%STATIC_GSIZE;
    __syncthreads();
    
    if(tid==0)
        atomicAdd(counts+gid,(int)s[d]);

    while(tid<d ){   
        atomicAdd(centroids+(gid*d+tid), s[tid]);
        tid+=BLOCKWIDTH;
    }
}

__global__ void cuda_centroid_means(point_t centroids, int* counts, int k, int d){
    int tid = threadIdx.x; int bid = blockIdx.x;

    if(tid>=d)return;

    while(tid<d){
        centroids[bid*d+tid]/=counts[bid];
        tid+=BLOCKWIDTH;
    }
    tid = threadIdx.x;

    // if(tid==0){
    //     printf("point %d: %f\n", bid, centroids[bid*d+tid]);
    // }
}