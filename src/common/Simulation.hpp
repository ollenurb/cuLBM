#pragma once

#include "SDL.h"
#include "D2Q9.hpp"

class Simulation {
protected:
  const unsigned int WIDTH;
  const unsigned int HEIGHT;

public:
  Simulation(unsigned int w, unsigned int h) : WIDTH(w), HEIGHT(h) {}

  virtual const D2Q9::LatticeNode *get_lattice() = 0;
  virtual void step() = 0;

  unsigned int get_width() const { return WIDTH; }
  unsigned int get_height() const { return HEIGHT; }

};
