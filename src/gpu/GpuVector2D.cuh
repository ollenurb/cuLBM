//
// Created by matteo on 7/18/21.
//
#pragma once

/* 2 dimensional vector */
template <typename T>
struct GpuVector2D {
    T x;
    T y;
    __host__ __device__ T mod_sqr() const;
    __host__ __device__ T modulus() const;
    template <typename U>
    __host__ __device__ inline double operator *(struct GpuVector2D<U> &v) const;
};

/* +============+ Vector2D methods +============+ */
/* (They are const because they do not modify member's state) */
template<typename T>
__host__ __device__ inline T GpuVector2D<T>::mod_sqr() const {
    return (x * x) + (y * y);
}

template<typename T>
__host__ __device__ inline T GpuVector2D<T>::modulus() const {
    return sqrt(mod_sqr());
}

template<typename T>
template<typename U>
__host__ __device__ inline double GpuVector2D<T>::operator*(GpuVector2D<U> &v) const {
    return (v.x * x) + (v.y * y);
}
