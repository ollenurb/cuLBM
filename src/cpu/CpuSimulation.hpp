#pragma once

#include <vector>
#include "SDL.h"
#include "../common/Matrix.cuh"
#include "../common/Vector2D.hpp"
#include "../common/Simulation.hpp"

/* Number of velocity vectors */
/* This class represents the 2DQ9 model */
#define Q 9
#define WEIGHTS {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}

/* A lattice Node
 * f[i] = f_i
 * density_eq[i] = f_eq_i (TODO: TO REMOVE)
 * u = u
 * rho = rho
 */
typedef struct LatticeNode {
  double density[Q] = WEIGHTS;
  double total_density = 1.0;
  Vector2D<double> macroscopic_velocity = {0, 0};
} LatticeNode;

class LBM : public Simulation {
private:
  /* +=========+ Constants +=========+ */
  const double VELOCITY = 0.070;
  const double VISCOSITY = 0.020;
  const double OMEGA = 1 / (3 * VISCOSITY + 0.5);

  /* Allowed displacement vectors */
  const Vector2D<int> e[Q] =
          {
                  {0,  0},
                  {1,  0},
                  {0,  1},
                  {-1, 0},
                  {0,  -1},
                  {1,  1},
                  {-1, 1},
                  {-1, -1},
                  {1,  -1}
          };

  /* Weights associated with each direction */
  const double W[Q] = WEIGHTS;

  /* +=========+ Variables +=========+ */
  Matrix<LatticeNode> lattice;
  Matrix<LatticeNode> lattice_t;
  LatticeNode initial_config;

  /* +=========+ LBM Steps +=========+ */
  void stream();

  void collide();

  void bounce();

public:
  LBM(unsigned int, unsigned int);

  ~LBM();

  /* Render the lattice state on the screen */
  void render(SDL_Texture *) override;

  /* Perform a simulation step: f(t) -> f(t + dt) */
  void step() override;
};
