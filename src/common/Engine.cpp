#include <iostream>
#include "Engine.hpp"

Engine::Engine(Simulation &r) : renderizable(r), WIDTH(r.get_width()), HEIGHT(r.get_height()) {
    running = false;
    SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, 0, &window, &renderer);
    screen = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
}

Engine::~Engine() = default;

void Engine::run()
{
    unsigned n_frame = 0;
    unsigned long long iterations = 0;
    running = true;
    while(running) {
        process_events();
        /* TODO: Change 10 with AFTER_NFRAMES */
        if(n_frame == 5) {
            renderizable.render(screen);
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, screen, NULL, NULL);
            SDL_RenderPresent(renderer);
            n_frame = 0;
            SDL_Delay(60);
            iterations++;
        }
        renderizable.step();
        n_frame++;
    }
    std::cout << "Simulations took " << iterations <<  " iterations" << std::endl;
}

void Engine::process_events()
{
    SDL_Event event;

    while(SDL_PollEvent(&event)) {
        switch(event.type) {
        case SDL_KEYDOWN:
            if(event.key.keysym.sym == SDLK_ESCAPE) {
                running = false;
            }
            break;
        case SDL_QUIT:
            running = false;
            break;
        }
    }
}
