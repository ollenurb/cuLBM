#pragma once

#include <vector>
#include "SDL.h"
#include "Matrix.hpp"
#include "Renderizable.hpp"

/* Number of velocity vectors */
/* This class represents the 2DQ9 model */
#define Q 9
#define D 2
#define WEIGHTS {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}
#define Ni 2
#define Si 4
#define Ei 1
#define Wi 3
#define NWi 6
#define NEi 5
#define SEi 7
#define SWi 8

typedef struct Vector2D {
    double x;
    double y;
    double mod_sqr() const;
    double modulus() const;
    double operator *(Vector2D &v) const;
} Vector2D;

// Initialize values with weights
typedef struct LatticeNode {
    double density[Q] = WEIGHTS;
    double density_eq[Q] = WEIGHTS;
    double total_density = 1.0;
    Vector2D macroscopic_velocity = {0, 0};
} LatticeNode;

class Lattice : public Renderizable
{
    private:
    /* +=========+ Constants +=========+ */
    const double VELOCITY = 0.070;
    const double VISCOSITY = 0.020;
    const double OMEGA = 1 / (3 * VISCOSITY + 0.5);

    /* Allowed displacement vectors */
    const Vector2D e[Q] =
    {
        { 0, 0}, { 1,  0}, {0,  1},
        {-1, 0}, { 0, -1}, {1,  1},
        {-1, 1}, {-1, -1}, {1, -1}
    };

    /* Weights associated with each direction */
    const double W[Q] = WEIGHTS;

    /* +=========+ Variables +=========+ */
    Matrix<LatticeNode> lattice;
    Matrix<LatticeNode> lattice_t;

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
