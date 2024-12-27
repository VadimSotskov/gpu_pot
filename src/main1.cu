#include <cuda_runtime.h>
#include <iostream>

struct Data {
    int *array;
    int size;
};

__global__ void printDynamicallyAllocatedMemory(int *d_array, int size) {
    for (int i = 0; i < size; ++i) {
        printf("Value[%d]: %d\n", i, d_array[i]);
    }
}

int main() {
    const int SIZE = 3;
    Data h_data;
    h_data.size = SIZE;
    h_data.array = new int[SIZE]{10, 20, 30};

    Data *d_data;
    int *d_array;
    cudaMalloc(&d_data, sizeof(Data));
    cudaMalloc(&d_array, SIZE * sizeof(int));
    cudaMemcpy(d_array, h_data.array, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Update h_data to point to device memory
    h_data.array = nullptr;
    cudaMemcpy(d_data, &h_data, sizeof(Data), cudaMemcpyHostToDevice);

    printDynamicallyAllocatedMemory<<<1, 1>>>(d_array, SIZE);
    cudaDeviceSynchronize();

    delete[] h_data.array;
    cudaFree(d_array);
    cudaFree(d_data);
    return 0;
}