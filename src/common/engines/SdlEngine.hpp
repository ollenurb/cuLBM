#pragma once

#include "../Simulation.hpp"
#include "SDL.h"

class SdlEngine {
private:
  const unsigned int WIDTH;
  const unsigned int HEIGHT;
  bool running;
  Simulation &simulation;
  SDL_Window *window;
  SDL_Renderer *renderer;
  SDL_Texture *screen;

  void process_events();
  void render(SDL_Texture*);

public:
  explicit SdlEngine(Simulation &);
  ~SdlEngine();
  void run();
};
