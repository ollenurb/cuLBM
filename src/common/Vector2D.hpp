//
// Created by matteo on 7/18/21.
//
#ifndef LBM_VECTOR2D_HPP
#define LBM_VECTOR2D_HPP

#include "Defines.hpp"
#include <cmath>

/* 2 dimensional vector */
template<typename T>
struct Vector2D {
  T x;
  T y;
  HOST_DEVICE T mod_sqr() const;

  HOST_DEVICE T modulus() const;

  template<typename U>
  HOST_DEVICE inline T operator*(const struct Vector2D<U> &v) const;
};

/* +============+ Vector2D methods +============+ */
/* (They are const because they do not modify the member's state) */
template<typename T>
HOST_DEVICE
inline T Vector2D<T>::mod_sqr() const {
  return (x * x) + (y * y);
}

template<typename T>
HOST_DEVICE
inline T Vector2D<T>::modulus() const {
  return sqrt(mod_sqr());
}

template<typename T>
template<typename U>
HOST_DEVICE
inline T Vector2D<T>::operator*(const struct Vector2D<U> &v) const {
  return (v.x * x) + (v.y * y);
}

#endif