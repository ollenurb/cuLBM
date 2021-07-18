#pragma once

#include "Simulation.hpp"
#include "SDL.h"

class Engine {
    private:
    const unsigned int WIDTH;
    const unsigned int HEIGHT;
    bool running;
    Simulation &renderizable;
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *screen;

    void process_events();

    public:
    Engine(Simulation&);
    ~Engine();
    void run();
};
