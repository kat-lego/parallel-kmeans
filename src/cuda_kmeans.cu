#include <iostream>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>


#define MAX_DISTANCE 1000000000

using namespace std;

/**
 * Inititialises @param points with random numbers between -50 and 50
 * 
 * @param points: pointer to flat 2d memory for the data
 * @param n: number of data points
 * @param d: dimention of the data points
 */
void init_test_points(float* points, int n, int d){
    srand(time(0));
    for(int i=0;i<n*d;i++){
        points[i] = -50+rand()%(100);
    }
}

/**
 * Inititialises @param points with random numbers between -50 and 50
 * 
 * @param points: pointer to flat 2d memory for the data
 * @param n: number of data points
 * @param d: dimention of the data points
 */
void print_points(float* points, int n, int d){
    string s(10*d, '=');
    cout<<s<<endl;
    for(int i=0; i<n; i++){
        printf("[");
        for(int j=0; j<d; j++){
            printf("%0.3f, ", points[i*d+j]);
        }
        printf("]\n");
    }
    cout<<s<<endl<<endl;

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define BLOCKWIDTH 512
#define STATIC_GSIZE 8
#define MAXUNITS 268435455

__global__ void cuda_compute_distances(float* sums, float* data, float* centroids, int k, int d);
__global__ void cuda_compute_argmins(int* argmin, float* sums, int k);

__global__ void cuda_clear(float* centroids, int size);
__global__ void cuda_centroid_sums(float* centroids, int* counts, float* data, int* assign, int k, int n, int d);
__global__ void cuda_centroid_means(float* centroids, int* counts, int k, int d);


void assign_to_centroids(int* assign, float* data, float* centroids, int k, int n, int d){
    //declare pointers for device memory
    float *dev_data, *dev_centroids, *dev_distances;
    int* dev_assign;


    //allocate device memory
    size_t size_data = n*d*sizeof(float);
    size_t size_centroids = k*d*sizeof(float);
    size_t size_assign = n*sizeof(int);
    size_t size_distances = n*k*sizeof(float);
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
        exit(1);
    }

    //Run the kernels
    cuda_compute_distances<<< nblocks,BLOCKWIDTH>>>(dev_distances, dev_data, dev_centroids, k, d);
    cuda_compute_argmins<<<n, BLOCKWIDTH>>>(dev_assign, dev_distances, k);

    // copy to host
    cudaMemcpy( assign, dev_assign, size_assign, cudaMemcpyDeviceToHost);

    //free memory
    cudaFree(dev_assign);
    cudaFree(dev_centroids);
    cudaFree(dev_data);
    cudaFree(dev_distances);

}

void update_centroids(int* assign, float* data, float* centroids, int k, int n, int d){

    //declare pointers for device memory
    float *dev_data, *dev_centroids;
    int *dev_assign, *dev_counts;

    //allocate device memory
    size_t size_data = n*d*sizeof(float);
    size_t size_centroids = k*d*sizeof(float);
    size_t size_assign = n*sizeof(int);
    size_t size_counts = k*sizeof(int);
    cudaMalloc( (void**)&dev_data, size_data  );
    cudaMalloc( (void**)&dev_centroids, size_centroids );
    cudaMalloc( (void**)&dev_assign,  size_assign );
    cudaMalloc( (void**)&dev_counts,  size_counts );
    
    //copy over the device memory
    cudaMemcpy( dev_data, data, size_data , cudaMemcpyHostToDevice);
    cudaMemcpy( dev_assign, assign, size_assign , cudaMemcpyHostToDevice);

    //centroid updates
    int nblocks = (k*d+BLOCKWIDTH-1)/BLOCKWIDTH;
    int nblocks2 = k*STATIC_GSIZE;
    cuda_clear<<<nblocks,BLOCKWIDTH>>>(dev_centroids, k*d);
    cuda_centroid_sums<<<nblocks2,BLOCKWIDTH, (d+1)*sizeof(float)>>>(dev_centroids, dev_counts, dev_data, dev_assign, k, n, d);
    cuda_centroid_means<<<k,BLOCKWIDTH>>>(dev_centroids, dev_counts, k, d);

    // copy to host
    cudaMemcpy( centroids, dev_centroids, size_centroids, cudaMemcpyDeviceToHost);

    // //free memory
    cudaFree(dev_assign);
    cudaFree(dev_centroids);
    cudaFree(dev_data);
    cudaFree(dev_counts);
}

/**
 * Perfoms kmeans with cuda and stores the converged centroids in @param centroids
 * 
 * @param data: pointer to flat 2d memory for the data
 * @param centroids: pointer to flat 2d memory for the centroids
 * @param k: number of centroids
 * @param n: number of data points
 * @param d: dimention of the data points
 */
void cuda_kmeans(float* data, float* centroids, int k, int n, int d, int iterations)
{

    int* assign = new int[n];
    int it = 0;

    //initialise the centroids to the first k data points
    for(int i=0;i<k*d;i++)centroids[i] = data[i];

    while(it<iterations){
        assign_to_centroids(assign, data, centroids, k, n, d);
        update_centroids(assign, data, centroids, k, n, d);
        it++;
    }

    delete[] assign;

}

int main( int argc, char **argv ) {

    int k = 2; // 2 centroids
    int n = 10; // 10 data points
    int d = 5; // 3-dimension
    int it = 10; // number of iterations

    float* data = new float[n*d];
    float* centroids = new float[k*d];

    init_test_points(data, n, d);

    cuda_kmeans(data, centroids, k, n, d, it);
    print_points(centroids, k, d);

    delete[] data;
    delete[] centroids;

    return 0;
}

/**
* Compute the distances between each data point and centroids. 
*/
__global__ void cuda_compute_distances(float* sums, float* data, float* centroids, int k, int d){
    
    //get the group id, block id and thread id.
    int tid = threadIdx.x; int bid = blockIdx.x%k;
    int gid = blockIdx.x/k;
    
    //shared memory array to store the terms from the euclidean distance
    __shared__ float ds[BLOCKWIDTH];
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
__global__ void cuda_compute_argmins(int* argmin, float* sums, int k){
    int tid = threadIdx.x; int bid = blockIdx.x;

    __shared__ float dmins[BLOCKWIDTH];
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

__global__ void cuda_clear(float* centroids, int size){
    int id = blockIdx.x*blockDim.x+ threadIdx.x;
    if(id>=size)return;
    centroids[id]=0;
}

__global__ void cuda_centroid_sums(float* centroids, int* counts, float* data, int* assign, int k, int n, int d){
    int tid = threadIdx.x; int bid = blockIdx.x%STATIC_GSIZE;
    int gid = blockIdx.x/STATIC_GSIZE;
    
    extern __shared__ float s[];
    
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

__global__ void cuda_centroid_means(float* centroids, int* counts, int k, int d){
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

