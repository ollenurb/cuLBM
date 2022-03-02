#pragma once

#include <vector>
#include "../common/D2Q9.hpp"
#include "../common/Simulation.hpp"

using namespace D2Q9;

class CpuSimulation : public Simulation {
private:
  /* +=========+ Variables +=========+ */
  Vector2D<Real> initial_config_u;
  Real initial_config_f[Q];
  /* +=========+ CpuSimulation Steps +=========+ */
  void stream();
  void collide();
  void bounce();

public:
  CpuSimulation(unsigned int, unsigned int);

  /* Perform a simulation step: f(t) -> f(t + dt) */
  void step() override;
};
