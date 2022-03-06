//
// Created by matteo on 3/3/22.
//
#pragma once

#include "../common/Lattice.hpp"

struct GpuLattice : Lattice {
  GpuLattice(unsigned w, unsigned h) : Lattice(w, h) {
    cudaMalloc(&_f, sizeof(Real) * WIDTH * HEIGHT * Q);
    cudaMalloc(&_u, sizeof(Vector2D<Real>) * WIDTH * HEIGHT);
    cudaMalloc(&_obstacle, sizeof(bool) * WIDTH * HEIGHT);
  }

  ~GpuLattice() {
    cudaFree(_f);
    cudaFree(_u);
    cudaFree(_obstacle);
  }
};