//
// Created by matteo on 7/17/21.
//
#pragma once

#include <iostream>
#include <stdexcept>
#include <iomanip>
#include "../common/Defines.hpp"

template<typename T>
class Matrix {
private:
  T *data;
  const unsigned int X;
  const unsigned int Y;
  const unsigned int SZ;

public:
  Matrix(unsigned int x, unsigned int y);

  ~Matrix();

  inline T &operator()(unsigned int i, unsigned int j);

  void swap(Matrix<T> &swappable);

  void print();
};

/* Methods Implementations */
template<typename T>
Matrix<T>::Matrix(unsigned int x, unsigned int y) : X(x), Y(y), SZ(x * y) {
  /* Allocate the matrix on the host */
  data = new T[SZ];
}

template<typename T>
Matrix<T>::~Matrix() {
  delete data;
}

template<typename T>
inline T &Matrix<T>::operator()(unsigned int i, unsigned int j) {
  unsigned int index = i * Y + j;
  return data[index];
}

template<typename T>
void Matrix<T>::swap(Matrix<T> &swappable) {
  T *tmp = swappable.data;
  swappable.data = data;
  data = tmp;
}

template<typename T>
void Matrix<T>::print() {
  std::cout << std::endl;
  for (int i = 0; i < X; i++) {
    for (int j = 0; j < Y; j++) {
      std::cout << char(156) << std::setprecision(7) << std::fixed
                << data[i * X + j].u.x << " ";
    }
    std::cout << std::endl;
  }
}
