//
// Created by matteo on 7/20/21.
//
#pragma once

#include "Vector2D.hpp"

#define Q 9
#define WEIGHTS {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}
#define E {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0,-1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}}

namespace D2Q9 {
/* A lattice Node where:
 * f[i] = lattice f
 * u = velocity vector
 * In the GPU-based implementation, it's slightly different so that
 * it can use CUDA-specific types and also exploit data coalescence
 */
#ifdef __CUDA_ARCH__
  typedef struct LatticeNode {
    Real f[Q] = WEIGHTS;
    Vector2D<Real> u = {0, 0};
    bool obstacle = false;
  } LatticeNode;
#else
  typedef struct LatticeNode {
    Real f[Q] = WEIGHTS;
    Vector2D<Real> u = {0.0f, 0.0f};
    bool obstacle = false;
  } LatticeNode;
#endif

  constexpr static const Vector2D<Real> VELOCITY = {0.070, 0.0};
  constexpr static const Real VISCOSITY = 0.020;
  constexpr static const Real OMEGA = 1 / (3 * VISCOSITY + 0.5);

/* Allowed displacement vectors */
  constexpr static const Vector2D<int> e[Q] = E;

/* Weights associated with each direction */
  const Real W[Q] = WEIGHTS;
}
