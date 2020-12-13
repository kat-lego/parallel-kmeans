#ifndef UTILS_H
#define UTILS_H

#define MAX_DISTANCE 100000000
#include <cuda_runtime.h>

typedef float element_t;
typedef element_t* point_t;

void print_points(point_t points, int n, int d);
void init_test_points(point_t points, int n, int d);

struct performance_t {
    float runtime1;
    float runtime2;
    float runtime3;
    void clear();
};

struct cudaTime_t{
    cudaEvent_t start, stop;
    void start_time();
    void stop_time(float& time);
};
#endif