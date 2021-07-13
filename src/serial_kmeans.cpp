#include <iostream>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <time.h>
#include <math.h>

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
    float sum = 0;
    float min;
    int min_centroid;

    for(int i=0; i<n; i++){
        min = MAX_DISTANCE;
        for( int j=0; j<k;j++){
            //START CALCULATING DISTANCE
            sum = 0;
            for(int l=0; l<d; l++){
                sum+= (data[i*d+l]-centroids[j*d+l])*(data[i*d+l]-centroids[j*d+l]);
            }
            sum = sqrt(sum);
            //END CALCULATING DISTANCE

            if(sum<=min){
                min = sum;
                min_centroid = j;   
            }

        }
        assign[i] = min_centroid;
    }
}

void update_centroids(int* assign, float* data, float* centroids, int k, int n, int d){
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


/**
 * Perfoms kmeans in serial and stores the converged centroids in @param centroids
 * 
 * @param data: pointer to flat 2d memory for the data
 * @param centroids: pointer to flat 2d memory for the centroids
 * @param k: number of centroids
 * @param n: number of data points
 * @param d: dimention of the data points
 */
void serial_kmeans(float* data, float* centroids, int k, int n, int d, int iterations)
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
    // print_points(data, n, d);

    serial_kmeans(data, centroids, k, n, d, it);
    print_points(centroids, k, d);

    delete[] data;
    delete[] centroids;

    return 0;
}

