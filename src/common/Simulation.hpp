#pragma once

#include "SDL.h"
#include "D2Q9.hpp"

class Simulation {
protected:
  const unsigned int WIDTH;
  const unsigned int HEIGHT;
  D2Q9::Lattice lattice;
  D2Q9::Lattice lattice_t;

public:
  Simulation(unsigned int w, unsigned int h) :
          WIDTH(w), HEIGHT(h), lattice(w, h), lattice_t(w, h) {}
  virtual ~Simulation() { }
  virtual void step() = 0;
  const D2Q9::Lattice *get_lattice() { return &lattice; }
  unsigned int get_width() const { return WIDTH; }
  unsigned int get_height() const { return HEIGHT; }
};
