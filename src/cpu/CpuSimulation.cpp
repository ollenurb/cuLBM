#include "CpuSimulation.hpp"
#include <algorithm>

#define index(x, y) x * HEIGHT + y

CpuSimulation::CpuSimulation(unsigned int w, unsigned int h) : Simulation(w, h) {
  // initialize lattices
  lattice = new LatticeNode[WIDTH * HEIGHT];
  lattice_t = new LatticeNode[WIDTH * HEIGHT];

  Real e_dp_u;
  /* Initialize the initial configuration */
  initial_config.u = VELOCITY;
  /* Assign each lattice with the equilibrium f */
  for (int i = 0; i < Q; i++) {
    e_dp_u = initial_config.u * e[i];
    initial_config.f[i] = W[i] * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * initial_config.u.mod_sqr()));
  }

  /* Assign the simulation lattices with initial configuration's values */
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      /* Initialize flow velocity */
      unsigned index = index(x, y);
      lattice[index] = lattice_t[index] = initial_config;

      /* TODO: To remove, its just to put a circle in the center */
      unsigned rel_x = WIDTH / 2 - x;
      unsigned rel_y = HEIGHT / 2 - y;
      double r = sqrt(rel_x * rel_x + rel_y * rel_y);
      if(r < std::min(WIDTH, HEIGHT) * 0.2) {
        lattice[index].obstacle = lattice_t[index].obstacle = true;
        for (int i = 0; i < Q; i++) {
          lattice[index].f[i] = lattice_t[index].f[i] = 0;
          lattice[index].u = lattice_t[index].u = {0, 0};
        }
      }
    }
  }

}

CpuSimulation::~CpuSimulation() = default;

inline unsigned clamp(unsigned val, unsigned low, unsigned high) {
  return std::min(std::max(val, low), high);
}

void CpuSimulation::stream() {
  /* Move the fluid to neighbouring sites */
  unsigned x_index, y_index;

  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      for (int i = 0; i < Q; i++) {
        x_index = clamp(x + e[i].x, 0, WIDTH - 1);
        y_index = clamp(y + e[i].y, 0, HEIGHT - 1);
        lattice_t[index(x_index, y_index)].f[i] = lattice[index(x, y)].f[i];
      }
    }
  }

  /* "CpuSimulation sites along the edges contain fluid that
   * is always assigned to have the equilibrium number
   * densities for some fixed f and velocity"
   * (Schroeder - CpuSimulation-Boltzmann Fluid Dynamics)
   */
  for (int x = 0; x < WIDTH; x++) {
    lattice_t[index(x, 0)] = initial_config;
    lattice_t[index(x, (HEIGHT-1))] = initial_config;
  }

  for (int y = 0; y < HEIGHT; y++) {
    lattice_t[index(0, y)] = initial_config;
    lattice_t[index((WIDTH-1), y)] = initial_config;
  }
}

void CpuSimulation::collide() {
  Real total_density;
  Real f_eq;
  Real e_dp_u;
  Vector2D<Real> new_u{};

  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      LatticeNode &cur_node = lattice[index(x, y)];
      if(!cur_node.obstacle) {
        /* Compute the total f of lattice site at position (x, y) */
        total_density = 0;
        new_u.x = new_u.y = 0;

        for (int i = 0; i < Q; i++) {
          total_density += cur_node.f[i];
          /* Accumulate the f inside each component of flow_velocity */
          new_u.x += static_cast<Real>(e[i].x) * cur_node.f[i]; // U_{x} component
          new_u.y += static_cast<Real>(e[i].y) * cur_node.f[i]; // U_{y} component
        }

        /* Compute average to get the actual value of flow_velocity */
        new_u.x = (total_density > 0) ? (new_u.x / total_density) : 0;
        new_u.y = (total_density > 0) ? (new_u.y / total_density) : 0;

        /* Compute densities at thermal equilibrium */
        /* Equation (8) */
        for (int i = 0; i < Q; i++) {
          e_dp_u = new_u * e[i];
          f_eq = (total_density * W[i]) * (1 + (3 * e_dp_u) + (static_cast<Real>(4.5) * (e_dp_u * e_dp_u)) - (static_cast<Real>(1.5) * new_u.mod_sqr()));
          cur_node.f[i] += OMEGA * (f_eq - cur_node.f[i]);
        }
        cur_node.u = new_u;
      }
    }
  }
}

void CpuSimulation::bounce() {
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      LatticeNode &cur_node = lattice_t[index(x, y)];
      if (cur_node.obstacle) {
        lattice_t[index((x+1), y    )].f[1] = cur_node.f[3];
        lattice_t[index(x    , (y+1))].f[2] = cur_node.f[4];
        lattice_t[index((x-1),  y   )].f[3] = cur_node.f[1];
        lattice_t[index(x    , (y-1))].f[4] = cur_node.f[2];
        lattice_t[index((x+1), (y+1))].f[5] = cur_node.f[7];
        lattice_t[index((x-1), (y+1))].f[6] = cur_node.f[8];
        lattice_t[index((x-1), (y-1))].f[7] = cur_node.f[5];
        lattice_t[index((x+1), (y-1))].f[8] = cur_node.f[6];

        for (int i = 1; i < Q; i++) {
          cur_node.f[i] = 0;
        }
      }
    }
  }
}

void CpuSimulation::step() {
  collide();
  stream();
  bounce();
  std::swap(lattice_t, lattice);
}

const D2Q9::LatticeNode *CpuSimulation::get_lattice() {
  return lattice;
}