#pragma once

#include <vector>
#include "../common/D2Q9.hpp"
#include "../common/Solver.hpp"
#include "CpuLattice.hpp"

using namespace D2Q9;

class CpuSolver : public Solver {
private:
  CpuLattice lattice;
  CpuLattice lattice_t;
  // TODO: Refactor
  Vector2D<Real> initial_config_u;
  Real initial_config_f[Q];

  void stream();

  void collide();

  void bounce();

public:
  CpuSolver(unsigned int, unsigned int);

  virtual Lattice * get_lattice() override;

  void step() override;
};
