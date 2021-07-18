//
// Created by matteo on 7/18/21.
//
#pragma once

/* 2 dimensional vector */
template <typename T>
struct Vector2D {
    double x;
    double y;
    double mod_sqr() const;
    double modulus() const;
    template <typename U>
    double operator *(struct Vector2D<U> &v) const;
};

/* +============+ Vector2D methods +============+ */
/* (They are const because they do not modify member's state) */
template<typename T>
inline double Vector2D<T>::mod_sqr() const {
    return (x * x) + (y * y);
}

template<typename T>
inline double Vector2D<T>::modulus() const {
    return sqrt(mod_sqr());
}

template<typename T>
template<typename U>
double Vector2D<T>::operator*(Vector2D<U> &v) const {
    return (v.x * x) + (v.y * y);
}
