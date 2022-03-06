//
// Created by matteo on 3/3/22.
//
#pragma once

#include "Defines.hpp"
#include "Vector2D.hpp"
#include "D2Q9.hpp"

struct Lattice {
  const unsigned WIDTH, HEIGHT;
  Real *_f;
  Vector2D<Real> *_u;
  bool *_obstacle;

  Lattice(unsigned w, unsigned h) : WIDTH(w), HEIGHT(h) {}

  ~Lattice() {}

  void swap(Lattice &lattice) {
    std::swap(this->_f, lattice._f);
    std::swap(this->_u, lattice._u);
  }

  /* f(x, y, i) <=> f(<x, y>, e_i) */
  HOST_DEVICE
  Real& f(unsigned x, unsigned y, unsigned i) {
    return _f[x + WIDTH * (y + HEIGHT * i)];
  }

  /* u(<x, y>) <=> \vec{u}(<x, y>) */
  HOST_DEVICE
  Vector2D<Real>& u(unsigned x, unsigned y) {
    return _u[x * HEIGHT + y];
  }

  HOST_DEVICE
  bool& obstacle(unsigned x, unsigned y) {
    return _obstacle[x * HEIGHT + y];
  }
};