//
// Created by matteo on 9/16/21.
//

#pragma once

#include "../Solver.hpp"
#include "../D2Q9.hpp"

class VtkEngine {
private:
  const unsigned int WIDTH;
  const unsigned int HEIGHT;
  const unsigned int STEPS;
  Solver &simulation;

  // VTK-related functions
  void write_header(FILE *) const;
  void write_data(FILE *file, Lattice* lattice) const;

public:
  explicit VtkEngine(Solver &, unsigned int);
  ~VtkEngine();
  void run();

};