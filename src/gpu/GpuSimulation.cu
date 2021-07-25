//
// Created by matteo on 7/18/21.
//
#include "GpuSimulation.cuh"
#include "../common/Utils.hpp"

/* Translate a 2D index to a 1D index */
#define index(x, y) x * device::HEIGHT + y

/* Global variables to store block and grid dim */
/* Thread blocks are squares therefore they'll have dimension BLOCK_DIMxBLOCK_DIM */
#define BLOCK_DIM 10
dim3 dim_block;
dim3 dim_grid;

/* ============================================================================================================== */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Device Variables and Constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* ============================================================================================================== */
namespace device {
  CONSTANT unsigned int WIDTH;
  CONSTANT unsigned int HEIGHT;
  CONSTANT LatticeNode INITIAL_CONFIG;
  CONSTANT float W[Q];
  CONSTANT Vector2D<int> e[Q];
}

/* ============================================================================================================== */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Device-Specific Code and Kernels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* ============================================================================================================== */

/* Initialize the device lattice using initial config's values */
__global__ void init_kernel(LatticeNode *lattice, LatticeNode *lattice_t) {
  unsigned x_i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y_i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned index = index(x_i, y_i);
  lattice[index] = lattice_t[index] = device::INITIAL_CONFIG; // Initialize both lattices

  /* TODO: TO REMOVE
   * Put a spike of f at the center */
  if (x_i > 10 && x_i < 20 && y_i > 10 && y_i < 20) {
    for (float &i : lattice[index].f)
      i += 0.02;
  }
}

__device__ inline unsigned clamp(unsigned val, unsigned l, unsigned h) {
  return min(max(val, l), h);
}

/* Stream the fluid */
__device__ void stream(LatticeNode *lattice, LatticeNode *lattice_t) {
  unsigned int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y_i = blockIdx.y * blockDim.y + threadIdx.y;

  /* Move the fluid to neighbouring sites */
  unsigned x_t, y_t;
  unsigned index, index_t;
  index = index(x_i, y_i);

  /* Stream away fluid on each lattice site */
  for (int i = 0; i < Q; i++) {
    x_t = clamp(x_i + device::e[i].x, 0, device::WIDTH - 1);
    y_t = clamp(y_i + device::e[i].y, 0, device::HEIGHT - 1);
    index_t = index(x_t, y_t);
    lattice_t[index_t].f[i] = lattice[index].f[i];
  }

  /* Handle boundaries */
  if (y_i == device::HEIGHT - 1 || y_i == 0 || x_i == device::WIDTH - 1 ||
      x_i == 0) {
    lattice_t[index] = device::INITIAL_CONFIG;
  }
}

/* Collide the fluid */
__device__ void collide(LatticeNode *lattice) {
  unsigned x_i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y_i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned index = x_i * device::HEIGHT + y_i;
  if (index > device::WIDTH * device::HEIGHT) {
    printf("ERROR! Index out of bounds");
  }

  float total_density;
  float f_eq;
  float e_dp_u;
  Vector2D<float> new_u{};
  LatticeNode &cur_node = lattice[index];

  /* Compute the total f of lattice site at position (x, y) */
  total_density = 0.0;
  new_u.x = new_u.y = 0.0;

  for (int i = 0; i < Q; i++) {
    total_density += cur_node.f[i];
    /* Accumulate the f inside each component of flow_velocity */
    new_u.x += device::e[i].x * cur_node.f[i]; // U_{x} component
    new_u.y += device::e[i].y * cur_node.f[i]; // U_{y} component
  }
  /* Compute average to get the actual value of flow_velocity */
  /* "Cast" to 0 if the velocity is negative */
  new_u.x = max(0.0, new_u.x / total_density);
  new_u.y = max(0.0, new_u.y / total_density);

  /* Compute densities at thermal equilibrium */
  /* Equation (8) */
  for (int i = 0; i < Q; i++) {
    e_dp_u = new_u * device::e[i];
    f_eq = total_density * device::W[i] *
           (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) -
            (1.5f * new_u.mod_sqr()));
    cur_node.f[i] += D2Q9::OMEGA * (f_eq - cur_node.f[i]);
  }
  cur_node.u = new_u;
}
/* Bounce back fluid on boundaries/obstacles */
__device__ void bounce() {

}

/* Perform a LBM step */
__global__ void step_kernel(LatticeNode *lattice, LatticeNode *lattice_t) {
  collide(lattice);
  __threadfence();
  stream(lattice, lattice_t);
  bounce();
}

/* ============================================================================================================== */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class Implementation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* ============================================================================================================== */
GpuSimulation::GpuSimulation(unsigned int w, unsigned int h) : Simulation(w,
                                                                          h) {
  SIZE = WIDTH * HEIGHT;
  dim_block = dim3(BLOCK_DIM, BLOCK_DIM);
  dim_grid = dim3(WIDTH / dim_block.x, HEIGHT / dim_block.y);
  /* Allocate space for lattice objects on the device */
  cudaMemcpyToSymbol(device::WIDTH, &WIDTH, sizeof(unsigned int));
  cudaMemcpyToSymbol(device::HEIGHT, &HEIGHT, sizeof(unsigned int));
  cudaMemcpyToSymbol(device::e, &D2Q9::e, sizeof(Vector2D<int>) * Q);
  cudaMemcpyToSymbol(device::W, &D2Q9::W, sizeof(float) * Q);

  cudaMalloc(&device_lattice, sizeof(LatticeNode) * SIZE);
  cudaMalloc(&device_lattice_t, sizeof(LatticeNode) * SIZE);

  host_lattice = new LatticeNode[SIZE];

  /* Initialize the initial configuration */
  float e_dp_u;
  LatticeNode tmp_init_conf;
  tmp_init_conf.u.x = D2Q9::VELOCITY;
  tmp_init_conf.u.y = 0;
  /* Assign each lattice with the equilibrium f */
  for (int i = 0; i < Q; i++) {
    e_dp_u = tmp_init_conf.u * D2Q9::e[i];
    tmp_init_conf.f[i] =
            D2Q9::W[i] * (1 + (3 * e_dp_u) + (4.5 * (e_dp_u * e_dp_u)) -
                          (1.5 * tmp_init_conf.u.mod_sqr()));
  }

  cudaMemcpyToSymbol(device::INITIAL_CONFIG, &tmp_init_conf,
                     sizeof(LatticeNode));
  /* Initialize */
  printf("Block: %d, %d, %d, Grid: %d, %d, %d\n", dim_block.x, dim_block.y,
         dim_block.z, dim_grid.x, dim_grid.y,
         dim_grid.z);
  init_kernel<<<dim_grid, dim_block>>>(device_lattice, device_lattice_t);
  cudaDeviceSynchronize();
}

GpuSimulation::~GpuSimulation() {
  cudaFree(device_lattice);
  cudaFree(device_lattice_t);
  delete host_lattice;
}

void GpuSimulation::render(SDL_Texture *screen) {
  cudaMemcpy(host_lattice, device_lattice, sizeof(LatticeNode) * SIZE,
             cudaMemcpyDeviceToHost);
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
      b = std::min(host_lattice[x * HEIGHT + y].u.modulus() * 4, 1.0f);
      *(dest + x) = utils::HSBtoRGB(0.5, 1.0, b);
    }
  }
  SDL_UnlockTexture(screen);
}

void GpuSimulation::step() {
  cudaDeviceSynchronize();
  step_kernel<<<dim_grid, dim_block>>>(device_lattice, device_lattice_t);
  cudaDeviceSynchronize();
  LatticeNode *tmp = device_lattice_t;
  device_lattice_t = device_lattice;
  device_lattice = tmp;
}
