#include "CpuSolver.hpp"
#include <algorithm>

CpuSolver::CpuSolver(unsigned int w, unsigned int h) : Solver(w, h), lattice(w, h), lattice_t(w, h) {
  Real e_dp_u;
  /* Initialize the initial configuration */
  initial_config_u = VELOCITY;
  /* Assign each lattice with the equilibrium f */
  for (int i = 0; i < Q; i++) {
    e_dp_u = initial_config_u * e[i];
    initial_config_f[i] = W[i] * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * initial_config_u.mod_sqr()));
  }

  /* Assign the simulation lattices with initial configuration's values */
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      /* Initialize flow velocity */
      lattice.u(x, y) = lattice_t.u(x, y) = initial_config_u;

      for (int i = 0; i < Q; i++) {
        lattice.f(x, y, i) = lattice_t.f(x, y, i) = initial_config_f[i];
      }

      /* TODO: To remove, its just to put a circle in the center */
      unsigned rel_x = WIDTH / 2 - x;
      unsigned rel_y = HEIGHT / 2 - y;
      double r = sqrt(rel_x * rel_x + rel_y * rel_y);
      if (r < std::min(WIDTH, HEIGHT) * 0.2) {
        lattice.obstacle(x, y) = lattice_t.obstacle(x, y) = true;
        lattice.u(x, y) = lattice_t.u(x, y) = {0, 0};
        for (int i = 0; i < Q; i++) {
          lattice.f(x, y, i) = lattice_t.f(x, y, i) = 0;
        }
      }
    }
  }
}

inline unsigned clamp(unsigned val, unsigned low, unsigned high) {
  return std::min(std::max(val, low), high);
}

void CpuSolver::stream() {
  /* Move the fluid to neighbouring sites */
  unsigned x_index, y_index;

  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      for (int i = 0; i < Q; i++) {
        x_index = clamp(x + e[i].x, 0, WIDTH - 1);
        y_index = clamp(y + e[i].y, 0, HEIGHT - 1);
        lattice_t.f(x_index, y_index, i) = lattice.f(x, y, i);
      }
    }
  }

  /* "CpuSimulation sites along the edges contain fluid that
   * is always assigned to have the equilibrium number
   * densities for some fixed f and velocity"
   * (Schroeder - CpuSimulation-Boltzmann Fluid Dynamics)
   */
  // TODO: Can be changed
  for (int x = 0; x < WIDTH; x++) {
    lattice_t.u(x, HEIGHT - 1) = lattice_t.u(x, 0) = initial_config_u;
    for (int i = 0; i < Q; i++) {
      lattice_t.f(x, 0, i) = lattice_t.f(x, HEIGHT - 1, i) = initial_config_f[i];
    }
  }

  for (int y = 0; y < HEIGHT; y++) {
    lattice_t.u(0, y) = lattice_t.u(WIDTH - 1, y) = initial_config_u;
    for (int i = 0; i < Q; i++) {
      lattice_t.f(0, y, i) = lattice_t.f(WIDTH - 1, y, i) = initial_config_f[i];
    }
  }
}

void CpuSolver::collide() {
  Real total_density;
  Real f_eq;
  Real e_dp_u;
  Vector2D<Real> new_u{};

  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      if (lattice.obstacle(x, y)) continue;
      /* Compute the total f of lattice site at position (x, y) */
      total_density = 0;
      new_u.x = new_u.y = 0;

      for (int i = 0; i < Q; i++) {
        Real current_direction = lattice.f(x, y, i);
        total_density += current_direction;
        /* Accumulate the f inside each component of flow_velocity */
        new_u.x += static_cast<Real>(e[i].x) * current_direction; // U_{x} component
        new_u.y += static_cast<Real>(e[i].y) * current_direction; // U_{y} component
      }

      /* Compute average to get the actual value of flow_velocity */
      new_u.x = (total_density > 0) ? (new_u.x / total_density) : 0;
      new_u.y = (total_density > 0) ? (new_u.y / total_density) : 0;

      /* Compute densities at thermal equilibrium */
      /* Equation (8) */
      for (int i = 0; i < Q; i++) {
        e_dp_u = new_u * e[i];
        f_eq = (total_density * W[i]) * (1 + (3 * e_dp_u) + (static_cast<Real>(4.5) * (e_dp_u * e_dp_u)) -
                                         (static_cast<Real>(1.5) * new_u.mod_sqr()));
        lattice.f(x, y, i) += OMEGA * (f_eq - lattice.f(x, y, i));
      }
      lattice.u(x, y) = new_u;
    }
  }
}

void CpuSolver::bounce() {
  for (int x = 1; x < WIDTH; x++) {
    for (int y = 1; y < HEIGHT; y++) {
      if (lattice_t.obstacle(x, y)) {
        lattice_t.f(x + 1, y, 1) = lattice_t.f(x, y, 3);
        lattice_t.f(x, y + 1, 2) = lattice_t.f(x, y, 4);
        lattice_t.f(x - 1, y, 3) = lattice_t.f(x, y, 1);
        lattice_t.f(x, y - 1, 4) = lattice_t.f(x, y, 2);
        lattice_t.f(x + 1, y + 1, 5) = lattice_t.f(x, y, 7);
        lattice_t.f(x - 1, y + 1, 6) = lattice_t.f(x, y, 8);
        lattice_t.f(x - 1, y - 1, 7) = lattice_t.f(x, y, 5);
        lattice_t.f(x + 1, y - 1, 8) = lattice_t.f(x, y, 6);

        for (int i = 1; i < Q; i++) {
          lattice_t.f(x, y, i) = 0;
        }
      }
    }
  }
}

void CpuSolver::step() {
  collide();
  stream();
  bounce();
  lattice_t.swap(lattice);
}

Lattice *CpuSolver::get_lattice() {
  return &lattice;
}
