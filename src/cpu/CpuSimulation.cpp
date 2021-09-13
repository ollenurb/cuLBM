#include "CpuSimulation.hpp"
#include "../common/Utils.hpp"
#include <algorithm>

CpuSimulation::CpuSimulation(unsigned int w, unsigned int h) : Simulation(w, h),
                                                               lattice(w, h),
                                                               lattice_t(w, h) {
  float e_dp_u;
  /* Initialize the initial configuration */
  initial_config.u.x = VELOCITY;
  initial_config.u.y = 0;
  /* Assign each lattice with the equilibrium f */
  for (int i = 0; i < Q; i++) {
    e_dp_u = initial_config.u * e[i];
    initial_config.f[i] = W[i] * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * initial_config.u.mod_sqr()));
  }
  /* Assign the simulation lattices with initial configuration's values */
  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      /* Initialize flow velocity */
      lattice(x, y) = lattice_t(x, y) = initial_config;
    }
  }

  /* Put a 10x10 block of higher f at the center of the lattice */
  /* TODO: Remove it, it's here just for testing purposes */
  int size = 10;
  unsigned x_center = WIDTH / 2 - (size / 2);
  unsigned y_center = HEIGHT / 2 - (size / 2);

  for (unsigned x = x_center; x < x_center + size; x++) {
    for (unsigned y = y_center; y < y_center + size; y++) {
      for (float &i : lattice(x, y).f) {
        i += .070;
      }
    }
  }
}

CpuSimulation::~CpuSimulation() = default;

void CpuSimulation::render(SDL_Texture *screen) {
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
      b = std::min(lattice(x, y).u.modulus() * 4, 1.0f);
      *(dest + x) = utils::HSBtoRGB(0.5, 1, b);
    }
  }
  SDL_UnlockTexture(screen);
}

inline unsigned clamp(unsigned val, unsigned low, unsigned high) {
  return std::min(std::max(val, low), high);
}

void CpuSimulation::stream() {
  /* Move the fluid to neighbouring sites */
  unsigned x_index, y_index;

  for (int y = 0; y < HEIGHT; y++) {
    for (int x = 0; x < WIDTH; x++) {
      for (int i = 0; i < Q; i++) {
        x_index = clamp(x + e[i].x, 0, WIDTH - 1);
        y_index = clamp(y + e[i].y, 0, HEIGHT - 1);
        lattice_t(x_index, y_index).f[i] = lattice(x, y).f[i];
      }
    }
  }

  /* "CpuSimulation sites along the edges contain fluid that
   * is always assigned to have the equilibrium number
   * densities for some fixed f and velocity"
   * (Schroeder - CpuSimulation-Boltzmann Fluid Dynamics)
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

void CpuSimulation::collide() {
  float total_density;
  float f_eq;
  float e_dp_u;
  Vector2D<float> new_u{};

  for (int x = 0; x < WIDTH; x++) {
    for (int y = 0; y < HEIGHT; y++) {
      LatticeNode &cur_node = lattice(x, y);
      /* Compute the total f of lattice site at position (x, y) */
      total_density = 0;
      new_u.x = new_u.y = 0;

      for (int i = 0; i < Q; i++) {
        total_density += cur_node.f[i];
        /* Accumulate the f inside each component of flow_velocity */
        new_u.x += static_cast<float>(e[i].x) * cur_node.f[i]; // U_{x} component
        new_u.y += static_cast<float>(e[i].y) * cur_node.f[i]; // U_{y} component
      }

      /* Compute average to get the actual value of flow_velocity */
      /* "Cast" to 0 if the velocity is negative */
      new_u.x = std::max(0.0f, new_u.x / total_density);
      new_u.y = std::max(0.0f, new_u.y / total_density);

      /* Compute densities at thermal equilibrium */
      /* Equation (8) */
      for (int i = 0; i < Q; i++) {
        e_dp_u = new_u * e[i];
        f_eq = total_density * W[i] * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * new_u.mod_sqr()));
        cur_node.f[i] += OMEGA * (f_eq - cur_node.f[i]);
      }
      cur_node.u = new_u;
    }
  }
}

void CpuSimulation::bounce() {
  // TODO: Implement bounce back
}

void CpuSimulation::step() {
  collide();
  stream();
  bounce();
  lattice.swap(lattice_t);
}
