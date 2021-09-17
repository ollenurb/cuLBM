//
// Created by matteo on 7/18/21.
//
#pragma once

#include "../common/Simulation.hpp"
#include "../common/Defines.hpp"
#include "../common/D2Q9.hpp"

using namespace D2Q9;

class GpuSimulation : public Simulation {
private:
  /* +=========+ Variables +=========+ */
  /* In this case we need more variables to hold both the host lattice and device lattice GPU references */
  LatticeNode *host_lattice;
  LatticeNode *device_lattice{};
  LatticeNode *device_lattice_t{};

  unsigned int SIZE;

public:
  /* +=========+ Constants +=========+ */
  GpuSimulation(unsigned int, unsigned int);

  ~GpuSimulation();

  const D2Q9::LatticeNode *get_lattice() override;

  /* Perform a simulation step: f(t) -> f(t + dt) */
  void step() override;

};

