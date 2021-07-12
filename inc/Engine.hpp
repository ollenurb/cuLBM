#pragma once

#include <Renderizable.hpp>
#include <SDL2/SDL.h>

class Engine {
    private:
    const unsigned int WIDTH;
    const unsigned int HEIGHT;
    bool running;
    Renderizable &renderizable;
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *screen;

    void process_events();

    public:
    Engine(Renderizable&);
    ~Engine();
    void run();
};
