#include <iostream>
#include <vector>
#include <stdio.h>

class CudaClass {
public:
  int x = 4;
  double* v;
  CudaClass(int size) {
    v = new double[size];
    for(int i = 0; i < size; ++i)
        v[i] = i;
  }
  ~CudaClass() {
    delete[] v;
  }
};

__global__ void func(CudaClass* c) {
    printf("x: %d\n", c->x);
    //for(int i = 0; i < size; ++i)
        //std::cout<<c->v[i]<<std::endl;
}


int main() {
  CudaClass c(8);
  CudaClass* d_c;
  cudaMalloc((void **)&d_c, sizeof(CudaClass));
  cudaMemcpy(d_c, &c, sizeof(CudaClass), cudaMemcpyHostToDevice);
  func<<<1,1>>>(d_c);
}

