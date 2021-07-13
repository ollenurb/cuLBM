#pragma once

#include <SDL2/SDL.h>
#include <Renderizable.hpp>
#include <Tensor3D.hpp>

/* Number of velocity vectors */
/* This class represents the 2DQ9 model */
#define Q 9
#define D 2
#define W0 4/9

class Lattice : public Renderizable
{
    private:
    /* +=========+ Constants +=========+ */
    const double VELOCITY = 0.070;
    const double VISCOSITY = 0.020;
    /* const double OMEGA = (1/3)*((1/VISCOSITY)+6); */
    const double OMEGA = 1 / (3 * VISCOSITY + 0.5);


    /* Allowed displacement vectors */
    const int e[Q][D] =
    {
        { 0, 0}, { 1,  0}, {0,  1},
        {-1, 0}, { 0, -1}, {1,  1},
        {-1, 1}, {-1, -1}, {1, -1}
    };

    /* Weights associated with each direction */
    const double W[Q] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

    /* +=========+ Variables +=========+ */
    Tensor3D<double> flow_velocity;
    Tensor3D<double> density;
    /* We need 2 3-tensors to run the simulation to keep the density of the
     * current and next step  (t and t') */
    Tensor3D<double> density_t;

    /* +=========+ LBM Steps +=========+ */
    void stream();
    void collide();
    double thermal_eq(unsigned int, unsigned int, unsigned int);

    public:
    Lattice(unsigned int, unsigned int);
    ~Lattice();
    void render(SDL_Texture*) override;
    /* Run a simulation step: f(t) -> f(t + dt) */
    void step();
};
