//
// Created by matteo on 7/17/21.
//
#pragma once
#include <iostream>
#include <stdexcept>
#include <iomanip>

template <typename T>
class GpuMatrix {
private:
    T *device_data;

public:
    __host__   GpuMatrix(unsigned int x, unsigned int y);
    __host__   ~GpuMatrix();
    __host__  __device__ inline T& operator()(unsigned int i, unsigned int j);
    __host__   inline GpuMatrix<T>& operator=(const GpuMatrix<T> &m);
    __device__ void swap(GpuMatrix& swappable);
    __host__   void print();

    const unsigned int X;
    const unsigned int Y;
    const unsigned int SZ;
};

/* Methods Implementations */
template<typename T>
__host__ GpuMatrix<T>::GpuMatrix(unsigned int x, unsigned int y) : X(x), Y(y), SZ(x * y)
{
    cudaMallocManaged((void**) &device_data, sizeof(T) * SZ);
}

template<typename T>
__host__ GpuMatrix<T>::~GpuMatrix()
{
    cudaFree(device_data);
}

template<typename T>
__host__ __device__ inline T& GpuMatrix<T>::operator()(unsigned int i, unsigned int j) {
    return device_data[i * Y + j];
}

template<typename T>
__device__ void GpuMatrix<T>::swap(GpuMatrix &swappable)
{
    T *tmp = swappable.device_data;
    swappable.device_data = device_data;
    device_data = tmp;
}

template<typename T>
__host__ void GpuMatrix<T>::print()
{
    std::cout << std::endl;
    for(int i = 0; i < X; i++) {
        for(int j = 0; j < Y; j++) {
            std::cout << char(156) << std::setprecision(7) << std::fixed << device_data[i * X + j].total_density << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
__host__ GpuMatrix<T> &GpuMatrix<T>::operator=(const GpuMatrix<T> &m) {
    if(this == &m) {
        return *this;
    }
    if(device_data != nullptr) {
        cudaFree(device_data);
    }
    cudaMemcpy(m, this, sizeof(GpuMatrix<T>), cudaMemcpyHostToDevice);
    return *this;
}