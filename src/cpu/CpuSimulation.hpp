#pragma once

#include <vector>
#include "SDL.h"
#include "Matrix.hpp"
#include "../common/D2Q9.hpp"
#include "../common/Simulation.hpp"

using namespace D2Q9;

class CpuSimulation: public Simulation {
private:
  /* +=========+ Variables +=========+ */
  Matrix<LatticeNode> lattice;
  Matrix<LatticeNode> lattice_t;
  LatticeNode initial_config;

  /* +=========+ CpuSimulation Steps +=========+ */
  void stream();

  void collide();

  void bounce();

public:
  CpuSimulation(unsigned int, unsigned int);

  ~CpuSimulation();

  /* Render the lattice state on the screen */
  void render_SDL(SDL_Texture *) override;

  /* Perform a simulation step: f(t) -> f(t + dt) */
  void step() override;
};
