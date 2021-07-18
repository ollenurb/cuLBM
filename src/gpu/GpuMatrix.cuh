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
    T *data;
    const unsigned int X;
    const unsigned int Y;
    const unsigned int SZ;

public:
    __host__ __device__ GpuMatrix(unsigned int x, unsigned int y);
    __host__ __device__ ~GpuMatrix();
    __host__ __device__ inline T& operator()(unsigned int i, unsigned int j);
    __host__ __device__ void swap(GpuMatrix& swappable);
    __host__ __device__ void print();
};

/* Methods Implementations */
template<typename T>
__host__ __device__ GpuMatrix<T>::GpuMatrix(unsigned int x, unsigned int y) : X(x), Y(y), SZ(x * y)
{
    cudaMalloc(&data, sizeof(T) * SZ);
}

template<typename T>
__host__ __device__ GpuMatrix<T>::~GpuMatrix() { cudaFree(data); }

template<typename T>
__host__ __device__ inline T& GpuMatrix<T>::operator()(unsigned int i, unsigned int j)
{
    unsigned int index = i * Y + j;
    if(index > SZ) {
        throw std::invalid_argument("GpuMatrix: Index out of bounds");
    }
    return data[index];
}

template<typename T>
__host__ __device__ void GpuMatrix<T>::swap(GpuMatrix &swappable)
{
    T *tmp = swappable.data;
    swappable.data = data;
    data = tmp;
}

template<typename T>
__host__ __device__ void GpuMatrix<T>::print()
{
    std::cout << std::endl;
    for(int i = 0; i < X; i++) {
        for(int j = 0; j < Y; j++) {
            std::cout << char(156) << std::setprecision(7) << std::fixed << data[i * X + j].density[0] << " ";
        }
        std::cout << std::endl;
    }
}




