//
// Created by matteo on 9/16/21.
//

#pragma once

#include "../Simulation.hpp"

class VtkEngine {
private:
  const unsigned int WIDTH;
  const unsigned int HEIGHT;
  const unsigned int STEPS;
  Simulation &simulation;

  // VTK-related functions
  void write_header(FILE *);

public:
  explicit VtkEngine(Simulation &, unsigned int);
  ~VtkEngine();
  void run();
};