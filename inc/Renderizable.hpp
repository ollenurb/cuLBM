#pragma once

#include <SDL2/SDL.h>

class Renderizable {
    protected:
    const unsigned int WIDTH;
    const unsigned int HEIGHT;

    public:
    Renderizable(unsigned int w, unsigned int h) : WIDTH(w), HEIGHT(h) { }
    virtual void render(SDL_Texture*) = 0;
    unsigned int get_width() { return WIDTH; }
    unsigned int get_height() { return HEIGHT; }

};
