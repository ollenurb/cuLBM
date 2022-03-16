//
// Created by matteo on 3/16/22.
//
#ifndef LBM_ARRAY_HPP
#define LBM_ARRAY_HPP

#include "Defines.hpp"
#include <cassert>
#include <cstdio>

template<typename T, int Dimensions, typename Allocation>
struct Array {};

template<typename T, typename Allocation>
struct Array<T, 2, Allocation> {
    unsigned w, h;
    T *data;

    HOST_DEVICE
    inline T &operator()(unsigned x, unsigned y) {
        return data[x * h + y];
    }
};

template<typename T, typename Allocation>
struct Array<T, 3, Allocation> {
    unsigned w, h, d;
    T *data;

    HOST_DEVICE
    inline T &operator()(unsigned x, unsigned y, unsigned z) {
        return data[x + w * (y + h * z)];// CHANGE
    }
};

template<typename T>
Array<T, 3, Host> malloc_host(unsigned w, unsigned h, unsigned d) {
    Array<T, 3, Host> array;
    array.w = w;
    array.h = h;
    array.d = d;
    array.data = new T[w * h * d];
    return array;
}

template<typename T>
Array<T, 2, Host> malloc_host(unsigned w, unsigned h) {
    Array<T, 2, Host> array;
    array.w = w;
    array.h = h;
    array.data = new T[w * h];
    return array;
}

template<typename T, int Dimensions>
void free_array(Array<T, Dimensions, Host>& array) {
    assert(array.data != nullptr);
    delete array.data;
    array.data = nullptr;
}

#endif