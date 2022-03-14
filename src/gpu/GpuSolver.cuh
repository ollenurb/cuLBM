//
// Created by matteo on 7/18/21.
//
#pragma once

#include "../common/Solver.hpp"
#include "../common/Defines.hpp"
#include "../common/D2Q9.hpp"

using namespace D2Q9;

class GpuSolver : public Solver {
private:
  /* In this case we need more variables to hold both the host lattice and device lattice GPU references */
  Lattice<Device> device_lattice;
  Lattice<Device> device_lattice_t;
  LatticeNode equilibrium_configuration;


  int get_device();

public:
  GpuSolver(unsigned w, unsigned h);

  ~GpuSolver();

  /* Perform a simulation step: f(t) -> f(t + dt) */
  void step() override;

  Lattice<Host> *get_lattice() override;

};

