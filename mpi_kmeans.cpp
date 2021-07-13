#include <stdio.h>
#include <math.h>
#include <mpi.h>

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

void assign_to_centroids(int* assign, float* data, float* centroids, int k, int n, int d){
    int id, num_p;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

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

void update_centroids(int* assign, float* data, float* centroids, int k, int n, int d){
    int id, num_p;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

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

/**
 * Perfoms kmeans in serial and stores the converged centroids in @param centroids
 * 
 * @param data: pointer to flat 2d memory for the data
 * @param centroids: pointer to flat 2d memory for the centroids
 * @param k: number of centroids
 * @param n: number of data points
 * @param d: dimention of the data points
 */
void mpi_kmeans(float* data, float* centroids, int k, int n, int d, int iterations){
    int id, num_p;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

    int* assign = new int[n];
    int it = 0;

    //create a type for a data point
    MPI_Datatype point_type;
    MPI_Type_contiguous(d, MPI_FLOAT, &point_type);
    MPI_Type_commit(&point_type);

    // initial centroids
    if(id==0){
        //initialise the centroids to the first k data points
        for(int i=0;i<k*d;i++)centroids[i] = data[i];
    }

    MPI_Bcast(centroids, k, point_type, 0, MPI_COMM_WORLD);

    while(it<iterations){

        assign_to_centroids(assign, data, centroids, k, n, d);
        
        // broadcast assign[] to each task
        int s=n/num_p;
        for(int i=0;i<num_p;i++){
            MPI_Bcast(assign+s*i, s, MPI_INT, i, MPI_COMM_WORLD);
        }
        
        update_centroids(assign, data, centroids, k, n, d);
        
        // broadcast centroids[] to each task
        s = (k>num_p)?k/num_p:1;
        int up = (s==1)?k:num_p;
        for(int i=0;i<up;i++){
            MPI_Bcast(centroids+s*i*d, s, point_type, i, MPI_COMM_WORLD);
        }
        
        it++;
    }

    // print_points(centroids, k, d);
    
    MPI_Type_free(&point_type);
    delete[] assign;
}

int main( int argc, char **argv ) {
    MPI_Init(&argc, &argv);

    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    int k = 2; // 2 centroids
    int n = 10; // 10 data points
    int d = 5; // 3-dimension
    int it = 10; // number of iterations

    float* data = new float[n*d];
    float* centroids = new float[k*d];

    init_test_points(data, n, d);
    
    mpi_kmeans(data, centroids, k, n, d, it);

    if(pid==0){
        print_points(centroids, k, d);   
    }

    delete[] data;
    delete[] centroids;

    MPI_Finalize();
    return 0;
}
