//
// Created by matteo on 7/20/21.
//
#pragma once
#include "Vector2D.hpp"

#define Q 9
#define WEIGHTS {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}

namespace D2Q9 {
/* A lattice Node
 * density[i] = f_i
 * macroscopic_velocity = u
 * total_density = rho
 */
typedef struct LatticeNode {
    double density[Q] = WEIGHTS;
    double total_density = 1.0;
    Vector2D<double> macroscopic_velocity = {0, 0};
} LatticeNode;

const double VELOCITY = 0.070;
const double VISCOSITY = 0.020;
const double OMEGA = 1 / (3 * VISCOSITY + 0.5);

/* Allowed displacement vectors */
const Vector2D<int> e[Q] =
        {
                { 0, 0}, { 1,  0}, {0,  1},
                {-1, 0}, { 0, -1}, {1,  1},
                {-1, 1}, {-1, -1}, {1, -1}
        };

/* Weights associated with each direction */
const double W[Q] = WEIGHTS;

}
