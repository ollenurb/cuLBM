//
// Created by matteo on 7/18/21.
//
#pragma once

#include "../common/Solver.hpp"
#include "../common/Defines.hpp"
#include "../common/D2Q9.hpp"
#include "../cpu/CpuLattice.hpp"
#include "GpuLattice.cuh"

using namespace D2Q9;

class GpuSolver : public Solver {
private:
  /* +=========+ Variables +=========+ */
  /* In this case we need more variables to hold both the host lattice and device lattice GPU references */
  CpuLattice host_lattice;
  GpuLattice device_lattice;
  GpuLattice device_lattice_t;
  int get_device();

public:
  /* +=========+ Constants +=========+ */
  GpuSolver(unsigned int, unsigned int);

  ~GpuSolver();

  /* Perform a simulation step: f(t) -> f(t + dt) */
  void step() override;


};

