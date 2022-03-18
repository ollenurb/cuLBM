#ifndef LBM_SDL_ENGINE_HPP
#define LBM_SDL_ENGINE_HPP

#include "../Solver.hpp"
#include "SDL.h"

class SdlEngine {
private:
    Parameters params;
    bool running;
    Solver &simulation;
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *screen;

    void process_events();
    void render(SDL_Texture *);

public:
    explicit SdlEngine(Solver &);
    ~SdlEngine();
    void run();
};

#endif