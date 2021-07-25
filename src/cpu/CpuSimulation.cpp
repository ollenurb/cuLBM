#include "CpuSimulation.hpp"
#include "../common/Utils.hpp"
#include <algorithm>

LBM::LBM(unsigned int w, unsigned int h) : Simulation(w, h), lattice(w, h),
                                           lattice_t(w, h) {
  double e_dp_u;
  /* Initialize the initial configuration */
  initial_config.macroscopic_velocity.x = VELOCITY;
  initial_config.macroscopic_velocity.y = 0;
  /* Assign each lattice with the equilibrium density */
  for (int i = 0; i < Q; i++) {
    e_dp_u = e[i] * initial_config.macroscopic_velocity;
    initial_config.density[i] =
            W[i] * (1 + (3 * e_dp_u) + (4.5 * (e_dp_u * e_dp_u)) -
                    (1.5 * initial_config.macroscopic_velocity.mod_sqr()));
  }
  /* Initialize the simulation lattices to the initial configuration */
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      /* Initialize flow velocity */
      lattice(x, y) = lattice_t(x, y) = initial_config;
    }
  }

  /* Put a 10x10 block of higher density at the center of the lattice */
  int size = 10;
  int x_center = WIDTH / 2 - (size / 2);
  int y_center = HEIGHT / 2 - (size / 2);

  for (int x = x_center; x < x_center + size; x++) {
    for (int y = y_center; y < y_center + size; y++) {
      for (double &i : lattice(x, y).density) {
        i += .070;
      }
    }
  }
}

LBM::~LBM() = default;

void LBM::render(SDL_Texture *screen) {
  /* From Stack Overflow: void **pixels is a pointer-to-a-pointer; these are
   * typically used (in this kind of context) where the data is of a pointer
   * type but memory management is handled by the function you call.
   */
  void *pixels;
  int pitch;
  Uint32 *dest;
  float b;

  if (SDL_LockTexture(screen, nullptr, &pixels, &pitch) < 0) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't lock texture: %s\n",
                 SDL_GetError());
  }

  for (int y = 0; y < HEIGHT; y++) {
    dest = (Uint32 *) ((Uint8 *) pixels + y * pitch);
    for (int x = 0; x < WIDTH; x++) {
      b = std::min(lattice(x, y).macroscopic_velocity.modulus() * 4, 1.0);
      *(dest + x) = utils::HSBtoRGB(0.5, 1.0, b);
    }
  }
  SDL_UnlockTexture(screen);
}

inline unsigned clamp(unsigned val, unsigned low, unsigned high) {
  return std::min(std::max(val, low), high);
}

void LBM::stream() {
  /* Move the fluid to neighbouring sites */
  unsigned x_index, y_index;

  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      for (int i = 0; i < Q; i++) {
        x_index = clamp(x + e[i].x, 0, WIDTH - 1);
        y_index = clamp(y + e[i].y, 0, HEIGHT - 1);
        lattice_t(x_index, y_index).density[i] = lattice(x, y).density[i];
      }
    }
  }

  /* "LBM sites along the edges contain fluid that
   * is always assigned to have the equilibrium number
   * densities for some fixed density and velocity"
   * (Schroeder - LBM-Boltzmann Fluid Dynamics)
   */
  for (int x = 0; x < WIDTH; x++) {
    lattice_t(x, 0) = initial_config;
    lattice_t(x, HEIGHT - 1) = initial_config;
  }

  for (int y = 0; y < HEIGHT; y++) {
    lattice_t(0, y) = initial_config;
    lattice_t(WIDTH - 1, y) = initial_config;
  }
}

void LBM::collide() {
  double total_density;
  double density_eq;
  double e_dp_u;
  double mod_u;

  Vector2D<double> u{};

  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      LatticeNode &cur_node = lattice(x, y);
      /* Compute the total density of lattice site at position (x, y) */
      total_density = 0.0;
      u.x = u.y = 0.0;

      for (int i = 0; i < Q; i++) {
        total_density += cur_node.density[i];
        /* Accumulate the density inside each component of flow_velocity */
        u.x += cur_node.density[i] * e[i].x; // U_{x} component
        u.y += cur_node.density[i] * e[i].y; // U_{y} component
      }

      /* Compute average to get the actual value of flow_velocity */
      /* "Cast" to 0 if the velocity is negative */
      u.x = std::max(0.0, u.x / total_density);
      u.y = std::max(0.0, u.y / total_density);

      /* Compute densities at thermal equilibrium */
      /* Equation (8) */
      for (int i = 0; i < Q; i++) {
        e_dp_u = e[i] * u;
        density_eq =
                total_density * W[i] *
                (1 + (3 * e_dp_u) + (4.5 * (e_dp_u * e_dp_u)) -
                 (1.5 * u.mod_sqr()));
        cur_node.density[i] += OMEGA * (density_eq - cur_node.density[i]);
      }
      cur_node.total_density = total_density;
      cur_node.macroscopic_velocity = u;
    }
  }
}

void LBM::bounce() {
  // TODO: Implement bounce back
}

void LBM::step() {
  collide();
  stream();
  bounce();
  lattice.swap(lattice_t);
}
