#pragma once

#include <vector>
#include "../common/D2Q9.hpp"
#include "../common/Simulation.hpp"

using namespace D2Q9;

class CpuSimulation: public Simulation {
private:
  /* +=========+ Variables +=========+ */
  LatticeNode* lattice;
  LatticeNode* lattice_t;
  LatticeNode initial_config;

  /* +=========+ CpuSimulation Steps +=========+ */
  void stream();
  void collide();
  void bounce();

public:
  CpuSimulation(unsigned int, unsigned int);
  ~CpuSimulation();
  /* Perform a simulation step: f(t) -> f(t + dt) */
  void step() override;
  /* Get the lattice reference */
  const D2Q9::LatticeNode *get_lattice() override;
};
