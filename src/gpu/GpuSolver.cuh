//
// Created by matteo on 7/18/21.
//
#ifndef LBM_GPU_SOLVER_CUH
#define LBM_GPU_SOLVER_CUH

#include "../common/Solver.hpp"
#include "../common/Defines.hpp"

class GpuSolver : public Solver {
private:
  /* In this case we need more variables to hold both the host lattice and device lattice GPU references */
  Lattice<Device> device_lattice;
  Lattice<Device> device_lattice_t;
  Bitmap<Device> device_obstacle;

  int get_device();

public:
  GpuSolver(Parameters params);

  ~GpuSolver();

  /* Perform a simulation step: f(t) -> f(t + dt) */
  void step() override;

  Lattice<Host>& get_lattice() override;

};

#endif