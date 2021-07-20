//
// Created by matteo on 7/20/21.
//
#pragma once

#include "Vector2D.hpp"

#define Q 9
#define WEIGHTS {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}
#define E {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0,-1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}}

namespace D2Q9 {
/* A lattice Node
 * f[i] = f_i
 * u = u
 * rho = rho
 */
    typedef struct LatticeNode {
        float f[Q] = WEIGHTS;
        Vector2D<float> u = {0, 0};
    } LatticeNode;

    constexpr static const float VELOCITY = 0.070;
    constexpr static const float VISCOSITY = 0.020;
    constexpr static const float OMEGA = 1 / (3 * VISCOSITY + 0.5);

/* Allowed displacement vectors */
    constexpr static const Vector2D<int> e[Q] = E;

/* Weights associated with each direction */
    const float W[Q] = WEIGHTS;

}
