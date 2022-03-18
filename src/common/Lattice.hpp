//
// Created by matteo on 3/14/22.
//
#ifndef LBM_LATTICE_HPP
#define LBM_LATTICE_HPP

#include "D2Q9.hpp"
#include "Array.hpp"

/* Define a LatticeNode */
struct LatticeNode {
    Real f[Q];
    Vector2D<Real> u;
};

/* Multidimensional vectors are going to be stored as 1D array so that memory access is improved */
template<typename Allocation>
struct Lattice {
    Array<Real, 3, Allocation> f;
    Array<Vector2D<Real>, 2, Allocation> u;
};

/* Declare specific functions */
Lattice<Host> malloc_host(unsigned w, unsigned h);

template<typename Allocation>
void free_lattice(Lattice<Allocation>& lattice) {
    free_array(lattice.u);
    free_array(lattice.f);
}


#endif