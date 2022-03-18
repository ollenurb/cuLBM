//
// Created by matteo on 3/16/22.
//

#ifndef LBM_GPU_UTILS_CUH
#define LBM_GPU_UTILS_CUH

#include "../common/Array.hpp"

template<typename T>
Array<T, 2, Device> malloc_device(unsigned w, unsigned h) {
    Array<T, 2, Device> array;
    array.w = w;
    array.h = h;
    cudaMalloc(&array.data, sizeof(T) * w * h);
    return array;
}

template<typename T>
Array<T, 3, Device> malloc_device(unsigned w, unsigned h, unsigned d) {
    Array<T, 3, Device> array;
    array.w = w;
    array.h = h;
    array.d = d;
    cudaMalloc(&array.data, sizeof(T) * w * h * d);
    return array;
}

/* Free resources */
template<typename T, int Dimensions>
void free_array(Array<T, Dimensions, Device> &array) {
    assert(array.data != nullptr);
    cudaFree(array.data);
    array.data = nullptr;
}

/*
 * Synchronization functions
 */
// Given a Host array, synchronize it with the given device array
template<typename T>
void host_to_device(const Array<T, 2, Host> &host, const Array<T, 2, Device> &device) {
    assert(device.w == host.w && device.h == host.h);
    cudaMemcpy(device.data, host.data, sizeof(T) * host.w * host.h, cudaMemcpyHostToDevice);
}

template<typename T>
void host_to_device(const Array<T, 3, Host> &host, const Array<T, 3, Device> &device) {
    assert(device.w == host.w && device.h == host.h && device.d == host.d);
    cudaMemcpy(device.data, host.data, sizeof(T) * host.w * host.h * host.d, cudaMemcpyHostToDevice);
}

/*
 * Given a device array, synchronize with host
 */
template<typename T>
void device_to_host(const Array<T, 2, Device> &device, const Array<T, 2, Host> &host) {
    assert(device.w == host.w && device.h == host.h);
    cudaMemcpy(host.data, device.data, sizeof(T) * host.w * host.h, cudaMemcpyDeviceToHost);
}

template<typename T>
void device_to_host(const Array<T, 3, Device> &device, const Array<T, 3, Host> &host) {
    assert(device.w == host.w && device.h == host.h && device.d == host.d);
    cudaMemcpy(host.data, device.data, sizeof(T) * host.w * host.h * host.d, cudaMemcpyDeviceToHost);
}

Lattice<Device> malloc_device(unsigned w, unsigned h) {
    Lattice<Device> result;
    result.f = malloc_device<Real>(w, h, Q);
    result.u = malloc_device<Vector2D<Real>>(w, h);
    return result;
}

#endif