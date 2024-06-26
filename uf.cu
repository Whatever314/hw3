#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <uf.h>


// Initialize the Union Find structure
// void init(UnionFind* uf) {
//     for (int i = 0; i < uf->rows; i++) {
//         for (int j = 0; j < uf->cols; j++) {
//             uf->p[i][j] = i * uf->cols + j;  // Set each pixel to its own set
//         }
//     }
// }

// Find the ancestor of x with path compression
// int find(UnionFind* uf, int x) {
//     if (uf->p[x / uf->cols][x % uf->cols] != x) {
//         uf->p[x / uf->cols][x % uf->cols] = find(uf, uf->p[x / uf->cols][x % uf->cols]);
//     }
//     return uf->p[x / uf->cols][x % uf->cols];
// }

// Merge the sets of two pixels
// void union_sets(UnionFind* uf, int x, int y) {
//     int rootX = find(uf, x);
//     int rootY = find(uf, y);
//     if (rootX != rootY) {
//         uf->p[rootY / uf->cols][rootY % uf->cols] = rootX;
//     }
// }

// Returns the root index of node n
__device__ unsigned Find(const int* s_buf, unsigned n) {
    while (s_buf[n]-1 != n) {
        n = s_buf[n]-1;
    }
    return n;
}

// 
__device__ unsigned FindAndCompress(int* s_buf, unsigned idx) {
    unsigned n = idx;
    while (s_buf[n]-1 != n) {
        n = s_buf[n]-1;
        s_buf[idx] = n+1;
    }
    return n;
}

// Merges the UFTrees of a and b, linking one root to the other
__device__ void Union(int* s_buf, unsigned a, unsigned b) {

    bool done;

    do {

        a = Find(s_buf, a);
        b = Find(s_buf, b);

        if (a < b) {
            int old = atomicMin(s_buf + b, a);
            done = (old == b);
            b = old;
        }
        else if (b < a) {
            int old = atomicMin(s_buf + a, b);
            done = (old == a);
            a = old;
        }
        else {
            done = true;
        }

    } while (!done); // the last write to s_buf of the thread happens under the condition that a or b is root, so it is safe

}


__global__ void InitLabeling(int* img, int* labels, int label_rows, int label_cols) {
    // label each tile. if it is a foreground tile, assign a unique label(raster index + 1) to it, otherwise assign 0
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < label_rows && col < label_cols) {
        unsigned label_index = row * label_cols + col;
        unsigned img_index = 4*row*label_cols + 2*col;
        // if (row == 0 && col < 6) {
        //     printf("row: %d, col: %d, val: %d, %d, %d, %d\n", row, col, img[img_index], img[img_index+1], img[img_index+2*label_cols], img[img_index+2*label_cols+1]);
        // }
        img[img_index] = (img[img_index] != 0);
        img[img_index+1] = (img[img_index+1] != 0);
        img[img_index+2*label_cols] = (img[img_index+2*label_cols] != 0);
        img[img_index+2*label_cols+1] = (img[img_index+2*label_cols+1] != 0);
        // if (row == 0 && col < 6) {
        //     printf("row: %d, col: %d, val: %d, %d, %d, %d\n", row, col, img[img_index], img[img_index+1], img[img_index+2*label_cols], img[img_index+2*label_cols+1]);
        // }

        labels[label_index] = (label_index+1)*(img[img_index] || img[img_index+1] || img[img_index+2*label_cols] || img[img_index+2*label_cols+1]);
        // if (row < 5 && col < 6) {
        //     printf("label_index: %d, row: %d, label: %d\n", label_index, row, labels[label_index]);
        // }
    }
}

__global__ void Merge(const int* img, int* labels, int label_rows, int label_cols) {

    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned label_index = row * label_cols + col;
    unsigned img_index = 4*row*label_cols + 2*col;

    bool tile[4] = { static_cast<bool>(img[img_index]), static_cast<bool>(img[img_index+1]), static_cast<bool>(img[img_index+2*label_cols]), static_cast<bool>(img[img_index+2*label_cols+1]) };

    if (label_index == 0) {
        printf("img: %d %d %d %d\n", img[img_index], img[img_index+1], img[img_index+2*label_cols], img[img_index+2*label_cols+1]);
        printf("tile: %d %d %d %d\n", tile[0], tile[1], tile[2], tile[3]);
    }

    if (row < label_rows && col < label_cols && labels[label_index]) {
        if (col > 0 && row > 0 && img[img_index - 2*label_cols - 1] && tile[0]) {
            Union(labels, label_index, label_index - label_cols - 1);
        }
        if (row > 0 && (img[img_index - 2*label_cols] || img[img_index - 2*label_cols+1]) && (tile[0] || tile[1])) {
            Union(labels, label_index, label_index - label_cols);
        }
        if (row > 0 && col + 1 < label_cols && img[img_index - 2*label_cols + 2] && tile[1]) {
            Union(labels, label_index, label_index - label_cols + 1);
        }
        if (col > 0 && (img[img_index-1] || img[img_index+2*label_cols-1]) && (tile[0] || tile[2])) {
            Union(labels, label_index, label_index - 1);
        }
    }
}

__global__ void Compression(int* labels, int label_rows, int label_cols) {

    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned label_index = row * label_cols + col;

    if (row < label_rows && col < label_cols) {
        FindAndCompress(labels, label_index);
    }
}


__global__ void FinalLabeling(int* img, const int* labels, int label_rows, int label_cols) {

    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned label_index = row * label_cols + col;
    unsigned img_index = 4*row * label_cols + 2*col;

    if (row < label_rows && col < label_cols) {
        img[img_index] = labels[label_index]*img[img_index];
        img[img_index+1] = labels[label_index]*img[img_index+1];
        img[img_index+2*label_cols] = labels[label_index]*img[img_index+2*label_cols];
        img[img_index+2*label_cols+1] = labels[label_index]*img[img_index+2*label_cols+1];
    }
}

int connected_components_labeling(UnionFind* uf) {

    dim3 grid_size, block_size;
    block_size = dim3(BLOCK_COLS, BLOCK_ROWS, 1);
    int cols = (uf->cols+1)/2;
    int rows = (uf->rows+1)/2;
    grid_size = dim3((cols + BLOCK_COLS - 1) / BLOCK_COLS, (rows + BLOCK_ROWS - 1) / BLOCK_ROWS, 1);
    int *d_img, *d_labels;
    // cudaError_t err = cudaSuccess;
    cudaMalloc(&d_img, 4*rows*cols*sizeof(int));
    for (int i = 0; i < uf->rows; i++) {
        cudaMemcpy(d_img + 2*i*cols, uf->image+i, uf->cols, cudaMemcpyHostToDevice);
        if (uf->cols % 2) {
            d_img[2*i*cols + cols - 1] = 0;
        }
    }
    if (uf->rows % 2) {
        for (int i = 0; i < cols; i++) {
            d_img[2*cols*uf->rows + i] = 0;
        }
    }
    cudaMalloc(&d_labels, rows*cols*sizeof(int));
    InitLabeling<<<grid_size, block_size>>>(d_img, d_labels, rows, cols);

    // int* h_labels = new int[rows*cols];
    // cudaMemcpy(h_labels, d_labels, rows*cols*sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 8; i++) {
    //     for (int j = 0; j < 8; j++) {
    //         printf("%d ", h_labels[i*cols + j]);
    //     }
    //     printf("\n");
    // }

    Merge<<<grid_size, block_size>>>(d_img, d_labels, rows, cols);

    // int* h_labels = new int[rows*cols];
    // cudaMemcpy(h_labels, d_labels, rows*cols*sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 8; i++) {
    //     for (int j = 0; j < 8; j++) {
    //         printf("%d ", h_labels[i*cols + j]);
    //     }
    //     printf("\n");
    // }

    Compression<<<grid_size, block_size>>>(d_labels, rows, cols);
    FinalLabeling<<<grid_size, block_size>>>(d_img, d_labels, rows, cols);

    for (int i = 0; i < uf->rows; i++) {
        cudaMemcpy(uf->image+i, d_img + 2*i*cols, uf->cols, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_img);
    cudaFree(d_labels);

    return 0;
}

// // Write the result of the Union Find to a file
// void writeResultToFile(UnionFind* uf, const char* filename) {
//     FILE* fp = fopen(filename, "w");
//     if (fp == NULL) {
//         perror("Error opening file");
//         exit(EXIT_FAILURE);
//     }

//     // First line: print the number of rows and columns
//     fprintf(fp, "%d %d\n", uf->rows, uf->cols);

//     // Starting from the second line, print the result of the Union Find
//     for (int i = 0; i < uf->rows; i++) {
//         for (int j = 0; j < uf->cols; j++) {
//             fprintf(fp, "%d ", uf->p[i][j]);
//         }
//     }

//     fclose(fp);
// }

// int main(int argc, char* argv[]) {
//     if (argc != 3) {
//         fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
//         return EXIT_FAILURE;
//     }

//     const char* input_filename = argv[1];
//     const char* output_filename = argv[2];

//     UnionFind uf;
//     int rows, cols;
//     FILE* input_file = fopen(input_filename, "r");
//     if (input_file == NULL) {
//         perror("Error opening input file");
//         return EXIT_FAILURE;
//     }

//     fscanf(input_file, "%d %d", &rows, &cols);

//     uf.p = (int**)malloc(rows * sizeof(int*));
//     uf.image = (int**)malloc(rows * sizeof(int*));
//     uf.rows = rows;
//     uf.cols = cols;

//     for (int i = 0; i < rows; i++) {
//         uf.p[i] = (int*)malloc(cols * sizeof(int));
//         uf.image[i] = (int*)malloc(cols * sizeof(int));
//     }

//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             fscanf(input_file, "%d", &uf.image[i][j]);
//         }
//     }

//     fclose(input_file);

//     connected_components_labeling(&uf);
//     writeResultToFile(&uf, output_filename);

//     for (int i = 0; i < rows; i++) {
//         free(uf.p[i]);
//         free(uf.image[i]);
//     }
//     free(uf.p);
//     free(uf.image);

//     return 0;
// }