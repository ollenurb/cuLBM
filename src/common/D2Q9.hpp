//
// Created by matteo on 7/20/21.
//
#pragma once

#include "Array.hpp"
#include "Vector2D.hpp"
#define Q 9
#define WEIGHTS \
    { 4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36 }
#define E                                                                              \
    {                                                                                  \
        {0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, { 1, -1 } \
    }

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

    /* Define a LatticeNode */
    struct LatticeNode {
        Real f[Q];
        Vector2D<Real> u;
    };

    /* Multidimensional vectors are going to be stored as 1D array so that memory access is improved */
    template<typename Allocation>
    struct Lattice {};

    template<>
    struct Lattice<Host> {
        Array<Real, Host> f;
        Array<Vector2D<Real>, Host> u;

        void init(unsigned w, unsigned h) {
            f.init(w * h * Q);
            u.init(w * h);
        }

        void free() {
            f.free();
            u.free();
        }
    };

#ifdef __NVCC__
    template<>
    struct Lattice<Device> {
        Array<Real, Device> f;
        Array<Vector2D<Real>, Device> u;

        void init(unsigned w, unsigned h) {
            f.init(w * h * Q);
            u.init(w * h);
        }

        void free() {
            f.free();
            u.free();
        }

    };
#endif

    /* Bitmap for obstacles */
} /* namespace D2Q9 */
