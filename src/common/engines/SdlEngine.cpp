#include <iostream>
#include "SdlEngine.hpp"
#include "../Utils.hpp"

#define UPDATE_STEPS 50

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
    if (n_frame == UPDATE_STEPS) {
      render(screen);
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

void SdlEngine::render(SDL_Texture *) {
  const D2Q9::LatticeNode *lattice = simulation.get_lattice();
  void *pixels;
  int pitch;
  Uint32 *dest;
  Real b;

  if (SDL_LockTexture(screen, nullptr, &pixels, &pitch) < 0) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't lock texture: %s\n", SDL_GetError());
  }

  for (int y = 0; y < HEIGHT; y++) {
    dest = (Uint32 *) ((Uint8 *) pixels + y * pitch);
    for (int x = 0; x < WIDTH; x++) {
      b = std::min(lattice[x * HEIGHT + y].u.modulus() * 3, static_cast<Real>(1));
      *(dest + x) = utils::HSBtoRGB(0.5, 1, b);
    }
  }
  SDL_UnlockTexture(screen);
}
