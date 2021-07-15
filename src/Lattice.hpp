#pragma once

#include <vector>
#include "SDL.h"
#include "Renderizable.hpp"
#include "Tensor3D.hpp"

/* Number of velocity vectors */
/* This class represents the 2DQ9 model */
#define Q 9
#define D 2
#define WEIGHTS {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}

typedef struct Vector2D {
    double x;
    double y;
} Vector2D;

// Initialize values with weights
typedef struct LatticeNode {
    double density[Q] = WEIGHTS;
    double density_eq[Q] = WEIGHTS;
    double total_density{};
    Vector2D macroscopic_velocity = {0, 0};
} LatticeNode;

class Lattice : public Renderizable
{
    private:
    /* +=========+ Constants +=========+ */
    const double VELOCITY = 0.70;
    const double VISCOSITY = 0.020;
    const double OMEGA = 1 / (3 * VISCOSITY + 0.5);


    /* Allowed displacement vectors */
    const int e[Q][D] =
    {
        { 0, 0}, { 1,  0}, {0,  1},
        {-1, 0}, { 0, -1}, {1,  1},
        {-1, 1}, {-1, -1}, {1, -1}
    };

    /* Weights associated with each direction */
    const double W[Q] = WEIGHTS;

    /* +=========+ Variables +=========+ */
    Tensor3D<double> flow_velocity;
    Tensor3D<double> density;
    /* We need 2 3-tensors to run the simulation to keep the density of the
     * current and next step  (t and t') */
    Tensor3D<double> density_t;

    std::vector<LatticeNode> lattice;

    /* +=========+ LBM Steps +=========+ */
    void stream();
    void collide();

    public:
    Lattice(unsigned int, unsigned int);
    ~Lattice();
    void render(SDL_Texture*) override;
    /* Perform a simulation step: f(t) -> f(t + dt) */
    void step() override;
};
