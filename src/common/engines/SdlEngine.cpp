#include <iostream>
#include "SdlEngine.hpp"

SdlEngine::SdlEngine(Simulation &r) : WIDTH(r.get_width()),
                                HEIGHT(r.get_height()), simulation(r) {
  running = false;
  SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, 0, &window, &renderer);
  screen = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
}

SdlEngine::~SdlEngine() = default;

void SdlEngine::run() {
  unsigned n_frame = 0;
  unsigned long long iterations = 0;
  running = true;
  while (running) {
    process_events();
    /* TODO: Change 10 with AFTER_NFRAMES */
    if (n_frame == 5) {
      simulation.render_SDL(screen);
      SDL_RenderClear(renderer);
      SDL_RenderCopy(renderer, screen, nullptr, nullptr);
      SDL_RenderPresent(renderer);
      n_frame = 0;
      SDL_Delay(60);
      iterations++;
    }
    simulation.step();
    n_frame++;
  }
  std::cout << "Simulations took " << iterations << " iterations" << std::endl;
}

void SdlEngine::process_events() {
  SDL_Event event;

  while (SDL_PollEvent(&event)) {
    switch (event.type) {
      case SDL_KEYDOWN:
        if (event.key.keysym.sym == SDLK_ESCAPE) {
          running = false;
        }
        break;
      case SDL_QUIT:
        running = false;
        break;
    }
  }
}
