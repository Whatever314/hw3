
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_COLS 16
#define BLOCK_ROWS 16

typedef struct {
    int** p;        // Each pixel's ancestor node
    int** image;    // Image data
    int rows;       // Number of rows in the image
    int cols;       // Number of columns in the image
} UnionFind;

// Returns the root index of node n
__device__ unsigned Find(const int* s_buf, unsigned n);

// 
__device__ unsigned FindAndCompress(int* s_buf, unsigned idx);

// Merges the UFTrees of a and b, linking one root to the other
__device__ void Union(int* s_buf, unsigned a, unsigned b);


__global__ void InitLabeling(const int* img, int* labels, int label_rows, int label_cols);

__global__ void Merge(const int* img, int* labels, int label_rows, int label_cols);

__global__ void Compression(int* labels, int label_rows, int label_cols);


__global__ void FinalLabeling(int* img, const int* labels, int label_rows, int label_cols);

int connected_components_labeling(UnionFind* uf);