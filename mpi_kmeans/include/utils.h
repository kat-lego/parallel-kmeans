#ifndef UTILS_H
#define UTILS_H

#define MAX_DISTANCE 100000000

typedef float element_t;
typedef element_t* point_t;

void print_points(point_t points, int n, int d);
void init_test_points(point_t points, int n, int d);

struct performance_t {
    double runtime1;
    double runtime2;
    double runtime3;
    void clear();
};

#endif