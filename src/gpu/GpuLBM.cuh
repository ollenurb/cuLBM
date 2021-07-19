//
// Created by matteo on 7/18/21.
//
#pragma once

#include "../common/Simulation.hpp"

/* Number of velocity vectors */
/* This class represents the 2DQ9 model */
#define Q 9
#define WEIGHTS {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}

/* A host_lattice Node
 * density[i] = f_i
 * macroscopic_velocity = u
 * total_density = rho
 */
//typedef struct LatticeNode {
//    double density[Q] = WEIGHTS;
//    double total_density = 1.0;
//    GpuVector2D<double> macroscopic_velocity = {0, 0};
//} LatticeNode;

typedef struct LatticeNode_t LatticeNode;

class GpuLBM : public Simulation
{
private:
    /* +=========+ Variables +=========+ */
    /* In this case we need more variables to hold both the host lattice and device lattice GPU references */
    LatticeNode* host_lattice{};
    LatticeNode* device_lattice{};
    LatticeNode* device_lattice_t{};

    unsigned BLOCK_DIM = 20;
    dim3 dim_block;
    dim3 dim_grid;
    unsigned int SIZE;

public:
    /* +=========+ Constants +=========+ */
    GpuLBM(unsigned int, unsigned int);
    ~GpuLBM();

    /* Render the host_lattice state on the screen */
    void render(SDL_Texture*) override;
    /* Perform a simulation step: f(t) -> f(t + dt) */
    void step() override;
};

