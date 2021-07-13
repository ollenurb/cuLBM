#include <Engine.hpp>

Engine::Engine(Renderizable &r) : renderizable(r), WIDTH(r.get_width()), HEIGHT(r.get_height()) {
    running = false;
    SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, 0, &window, &renderer);
    screen = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
}

Engine::~Engine() { }

void Engine::run()
{
    unsigned n_frame = 0;
    running = true;
    while(running) {
        process_events();
        renderizable.step();

        /* TODO: Change 10 with AFTER_NFRAMES */
        if(n_frame == 1) {
            renderizable.render(screen);
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, screen, NULL, NULL);
            SDL_RenderPresent(renderer);
            n_frame = 0;
            SDL_Delay(60);
        }

        n_frame++;
    }
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
