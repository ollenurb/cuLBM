//
// Created by matteo on 7/20/21.
//
#pragma once

#include "Vector2D.hpp"
#define Q 9
#define WEIGHTS {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}
#define E {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0,-1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}}

namespace D2Q9 {
  /* Define LBM constants */
  constexpr static const Vector2D<Real> VELOCITY = {0.070, 0.0};
  constexpr static const Real VISCOSITY = 0.020;
  constexpr static const Real OMEGA = 1 / (3 * VISCOSITY + 0.5);

  /* Allowed displacement vectors */
  /* "cast" into Vector2D so that it is more convenient to use later */
  constexpr static const Vector2D<int> e[Q] = E;

  /* Weights associated to each direction */
  const Real W[Q] = WEIGHTS;
}
