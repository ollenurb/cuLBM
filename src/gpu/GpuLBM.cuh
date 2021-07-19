//
// Created by matteo on 7/18/21.
//
#pragma once

#include "../common/Simulation.hpp"
#include "GpuVector2D.cuh"
#include "GpuMatrix.cuh"

/* Number of velocity vectors */
/* This class represents the 2DQ9 model */
#define Q 9
#define WEIGHTS {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}

/* A host_lattice Node
 * density[i] = f_i
 * macroscopic_velocity = u
 * total_density = rho
 */
typedef struct LatticeNode {
    double density[Q] = WEIGHTS;
    double total_density = 1.0;
    GpuVector2D<double> macroscopic_velocity = {0, 0};
} LatticeNode;

class GpuLBM : public Simulation
{
private:
    /* +=========+ Constants +=========+ */
    const double VELOCITY = 0.070;
    const double VISCOSITY = 0.020;
    const double OMEGA = 1 / (3 * VISCOSITY + 0.5);

    /* Allowed displacement vectors */
    const GpuVector2D<int> e[Q] =
            {
                    { 0, 0}, { 1,  0}, {0,  1},
                    {-1, 0}, { 0, -1}, {1,  1},
                    {-1, 1}, {-1, -1}, {1, -1}
            };

    /* Weights associated with each direction */
    const double W[Q] = WEIGHTS;

    /* +=========+ Variables +=========+ */
    /* In this case we need more variables to hold both the host lattice and device lattice GPU references */
    GpuMatrix<LatticeNode> host_lattice;
    GpuMatrix<LatticeNode> host_lattice_t;
    GpuMatrix<LatticeNode>* device_lattice;
    GpuMatrix<LatticeNode>* device_lattice_t;
    LatticeNode host_initial_config;
    LatticeNode* device_initial_config;

public:
    GpuLBM(unsigned int, unsigned int);
    ~GpuLBM();

    /* Render the host_lattice state on the screen */
    void render(SDL_Texture*) override;
    /* Perform a simulation step: f(t) -> f(t + dt) */
    void step() override;
};

