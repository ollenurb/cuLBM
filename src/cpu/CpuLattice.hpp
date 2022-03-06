//
// Created by matteo on 3/3/22.
//
#pragma once

#include "../common/Lattice.hpp"

struct CpuLattice : Lattice {
  CpuLattice(unsigned w, unsigned h) : Lattice(w, h) {
    _f = new Real[w * h * Q];
    _u = new Vector2D<Real>[w * h];
    _obstacle = new bool[w * h];
  }

  ~CpuLattice() {
    delete _f;
    delete _u;
    delete _obstacle;
  }
};