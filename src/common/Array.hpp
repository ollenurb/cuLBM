//
// Created by matteo on 3/14/22.
//
#pragma once

#include "Defines.hpp"
#include <cstdlib>

template <typename Type, typename Allocation>
struct Array {};

template <typename Type>
struct Array<Type, Host> {
    Type *data;

    inline void init(size_t size) { data = new Type[size]; }

    inline void free() { delete data; }

    inline Type& operator [] (size_t index) { return data[index]; }


};

#ifdef __NVCC__
template <typename Type>
struct Array<Type, Device> {
    Type *data;

    HOST
    inline void init(size_t size) {
        cudaMalloc(&data, sizeof(Type) * size);
    }

    inline void free() {
        cudFree(data);
    }

    DEVICE
    inline Type& operator [] (size_t index) { return data[index]; }

    HOST
    inline int copy_to(Array<Type, Host> &array, size_t size) { cudaMemcpy(array.data, data, sizeof(size), cudaMemcpyDeviceToHost); }

    HOST
    inline int copy_from(Array<Type, Host> &array) { cudaMemcpy(data, array.data, sizeof(size), cudaMemcpyHostToDevice); }
};
#endif
