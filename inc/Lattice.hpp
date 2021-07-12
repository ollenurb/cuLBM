#pragma once

#include <SDL2/SDL.h>
#include <Renderizable.hpp>

/* Number of velocity vectors */
/* This class represents the 2DQ9 model */
#define Q 9

class Lattice : public Renderizable {
    private:

    public:
    Lattice(unsigned int, unsigned int);
    ~Lattice();
    void render(SDL_Texture*) override;
};
