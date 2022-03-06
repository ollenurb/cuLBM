#pragma once

#include "SDL.h"
#include "D2Q9.hpp"
#include "Lattice.hpp"

class Solver {
protected:
  const unsigned int WIDTH;
  const unsigned int HEIGHT;

public:
  Solver(unsigned int w, unsigned int h) : WIDTH(w), HEIGHT(h) {}

  virtual ~Solver() {}

  virtual void step() = 0;

  virtual Lattice *get_lattice() = 0;

  unsigned int get_width() const { return WIDTH; }

  unsigned int get_height() const { return HEIGHT; }
};