//
// Created by matteo on 3/16/22.
//
#include "Lattice.hpp"

Lattice<Host> malloc_host(unsigned w, unsigned h) {
    Lattice<Host> result;
    result.f = malloc_host<Real>(w, h, Q);
    result.u = malloc_host<Vector2D<Real>>(w, h);
    return result;
}