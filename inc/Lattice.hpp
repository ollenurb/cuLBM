#pragma once

#include <SDL2/SDL.h>
#include <Renderizable.hpp>
#include <Tensor3D.hpp>

/* Number of velocity vectors */
/* This class represents the 2DQ9 model */
#define Q 9
#define D 2

class Lattice : public Renderizable
{
    private:
    Tensor3D<double> flow_velocity;

    const float viscosity = 0.02;
    const float omega = (1/3)*((1/viscosity)+6);

    public:
    Lattice(unsigned int, unsigned int);
    ~Lattice();
    void render(SDL_Texture*) override;
};
