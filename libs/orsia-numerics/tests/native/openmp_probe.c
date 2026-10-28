#include <omp.h>

int default_team_size(void) {
    int size = 0;
    #pragma omp parallel reduction(+:size)
    size += 1;
    return size;
}

int explicit_team_size(void) {
    int size = 0;
    #pragma omp parallel num_threads(4) reduction(+:size)
    size += 1;
    return size;
}

int nested_team_size(void) {
    int size = 0;
    #pragma omp parallel num_threads(4) reduction(+:size)
    {
        int inner = 0;
        #pragma omp parallel num_threads(4) reduction(+:inner)
        inner += 1;
        size += inner;
    }
    return size;
}
