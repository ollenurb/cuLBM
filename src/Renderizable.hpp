#pragma once

#include "SDL.h"

class Renderizable {
    protected:
    const unsigned int WIDTH;
    const unsigned int HEIGHT;

    public:
    Renderizable(unsigned int w, unsigned int h) : WIDTH(w), HEIGHT(h) { }

    /* IMPROVEMENT: It is possible to improve this API by abstracting away the
     * implementation details.  SDL_Texture is basically a canvas on which the
     * user is going to write on. It can be replaced by a Matrix of dimension
     * [width][height] of colors. Colors can be represented as a 32bit unsigned
     * int numbers. In this way, the Engine will be the only one responsible
     * for calling the SDL API */
    virtual void render(SDL_Texture*) = 0;
    virtual void step() = 0;
    unsigned int get_width() { return WIDTH; }
    unsigned int get_height() { return HEIGHT; }

};
