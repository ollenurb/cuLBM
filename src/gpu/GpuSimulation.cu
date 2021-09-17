//
// Created by matteo on 7/18/21.
//
#include "GpuSimulation.cuh"
#include <algorithm>

/* Translate a 2D index to a 1D index */
#define index(x, y) x * device::HEIGHT + y

/* Global variables to store block and grid dim */
/* Thread blocks are squares therefore they'll have dimension BLOCK_DIMxBLOCK_DIM */
#define BLOCK_DIM 10
dim3 dim_block;
dim3 dim_grid;

__device__ void compute_macroscopics(const LatticeNode& node, Vector2D<Real> &new_u, Real &rho);

/* ============================================================================================================== */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Device Variables and Constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* ============================================================================================================== */
namespace device {
  CONSTANT unsigned int WIDTH;
  CONSTANT unsigned int HEIGHT;
  CONSTANT LatticeNode INITIAL_CONFIG;
  CONSTANT Real W[Q];
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

  /* TODO: TO REMOVE Put a circle at the center of the simulation */
  unsigned rel_x = device::WIDTH / 2 - x_i;
  unsigned rel_y = device::HEIGHT / 2 - y_i;
  double r = sqrt(static_cast<float>(rel_x * rel_x + rel_y * rel_y));

  if(r < min(device::WIDTH, device::HEIGHT) * 0.05) {
    lattice[index].obstacle = lattice_t[index].obstacle = true;
    lattice[index].u = lattice_t[index].u = {0, 0};
    for (int i = 0; i < Q; i++) {
      lattice[index].f[i] = lattice_t[index].f[i] = 0;
    }
  }
}

/* Stream the fluid */
__global__ void stream(LatticeNode *lattice, LatticeNode *lattice_t) {
  unsigned int x_i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y_i = blockIdx.y * blockDim.y + threadIdx.y;

  /* Move the fluid to neighbouring sites */
  unsigned x_t, y_t;
  unsigned index, index_t;
  index = index(x_i, y_i);


  if (index < device::WIDTH * device::HEIGHT) {
    /* Stream away fluid on each lattice site */
    for (int i = 0; i < Q; i++) {
      x_t = x_i + device::e[i].x;
      y_t = y_i + device::e[i].y;
      index_t = index(x_t, y_t);

      if (x_t >= 0 && y_t >= 0 && x_t < device::WIDTH && y_t < device::HEIGHT) {
        lattice_t[index_t].f[i] = lattice[index].f[i];
      }
    }

    /* Handle boundaries */
    if (y_i == device::HEIGHT - 1 || y_i == 0 || x_i == device::WIDTH - 1 || x_i == 0) {
      lattice_t[index] = device::INITIAL_CONFIG;
    }
  }
}

/* Collide the fluid according to LBM equilibrium density */
__global__ void collide(LatticeNode *lattice) {
  unsigned x_i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y_i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned index = index(x_i, y_i);

  Real total_density;
  Real f_eq;
  Real e_dp_u;
  Vector2D<Real> new_u{};
  LatticeNode &cur_node = lattice[index];

  if(!cur_node.obstacle) {
    compute_macroscopics(cur_node, new_u, total_density);

    /* Compute densities at thermal equilibrium */
    for (int i = 0; i < Q; i++) {
      e_dp_u = new_u * device::e[i];
      f_eq = (total_density * device::W[i]) * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * new_u.mod_sqr()));
      cur_node.f[i] += D2Q9::OMEGA * (f_eq - cur_node.f[i]);
    }
    cur_node.u = new_u;
  }
}

__device__ void compute_macroscopics(const LatticeNode& node, Vector2D<Real> &new_u, Real &rho) {
  // Prepare values
  rho = 0;
  new_u = {0, 0};

  for (int i = 0; i < Q; i++) {
    rho += node.f[i];
    /* Accumulate the f inside each component of flow_velocity */
    new_u.x += device::e[i].x * node.f[i]; // U_{x} component
    new_u.y += device::e[i].y * node.f[i]; // U_{y} component
  }
  /* Compute average to get the actual value of flow_velocity */
  /* "Cast" to 0 if the velocity is negative */
  new_u.x = (rho > 0) ? (new_u.x / rho) : 0;
  new_u.y = (rho > 0) ? (new_u.y / rho) : 0;
}

/* Bounce back fluid on obstacles */
__global__ void bounce(LatticeNode *lattice_t) {
  unsigned x_i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y_i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned index = index(x_i, y_i);
  LatticeNode &cur_node = lattice_t[index];

  /* Sadly, lots of threads are going to diverge here */
  if (cur_node.obstacle) {
    lattice_t[index((x_i + 1), y_i)].f[1] = cur_node.f[3];
    lattice_t[index(x_i, (y_i + 1))].f[2] = cur_node.f[4];
    lattice_t[index((x_i - 1), y_i)].f[3] = cur_node.f[1];
    lattice_t[index(x_i, (y_i - 1))].f[4] = cur_node.f[2];
    lattice_t[index((x_i + 1), (y_i + 1))].f[5] = cur_node.f[7];
    lattice_t[index((x_i - 1), (y_i + 1))].f[6] = cur_node.f[8];
    lattice_t[index((x_i - 1), (y_i - 1))].f[7] = cur_node.f[5];
    lattice_t[index((x_i + 1), (y_i - 1))].f[8] = cur_node.f[6];

    for(int i = 1; i < Q; i++) {
      cur_node.f[i] = 0;
    }
  }
}

/* ============================================================================================================== */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class Implementation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* ============================================================================================================== */
GpuSimulation::GpuSimulation(unsigned int w, unsigned int h) : Simulation(w, h) {
  SIZE = WIDTH * HEIGHT;
  dim_block = dim3(BLOCK_DIM, BLOCK_DIM);
  dim_grid = dim3(WIDTH / dim_block.x, HEIGHT / dim_block.y);
  /* Allocate space for lattice objects on the device */
  cudaMemcpyToSymbol(device::WIDTH, &WIDTH, sizeof(unsigned int));
  cudaMemcpyToSymbol(device::HEIGHT, &HEIGHT, sizeof(unsigned int));
  cudaMemcpyToSymbol(device::e, &D2Q9::e, sizeof(Vector2D<int>) * Q);
  cudaMemcpyToSymbol(device::W, &D2Q9::W, sizeof(Real) * Q);

  cudaMalloc(&device_lattice, sizeof(LatticeNode) * SIZE);
  cudaMalloc(&device_lattice_t, sizeof(LatticeNode) * SIZE);

  host_lattice = new LatticeNode[SIZE];

  /* Compute the initial configuration's parameters */
  Real e_dp_u;
  LatticeNode tmp_init_conf;
  tmp_init_conf.u = D2Q9::VELOCITY;
  /* Assign each lattice with the equilibrium f */
  for (int i = 0; i < Q; i++) {
    e_dp_u = tmp_init_conf.u * D2Q9::e[i];
    tmp_init_conf.f[i] = D2Q9::W[i] * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * tmp_init_conf.u.mod_sqr()));
  }

  cudaMemcpyToSymbol(device::INITIAL_CONFIG, &tmp_init_conf, sizeof(LatticeNode));
  /* Initialize */
  printf("Block: %d, %d, %d, Grid: %d, %d, %d\n", dim_block.x, dim_block.y, dim_block.z, dim_grid.x, dim_grid.y, dim_grid.z);

  init_kernel<<<dim_grid, dim_block>>>(device_lattice, device_lattice_t);
  cudaDeviceSynchronize();
}

GpuSimulation::~GpuSimulation() {
  cudaFree(device_lattice);
  cudaFree(device_lattice_t);
  delete host_lattice;
}

void GpuSimulation::step() {
  collide<<<dim_grid, dim_block>>>(device_lattice);
  stream<<<dim_grid, dim_block>>>(device_lattice, device_lattice_t);
  bounce<<<dim_grid, dim_block>>>(device_lattice_t);
  cudaDeviceSynchronize();
  /* Swap pointers */
  std::swap(device_lattice_t, device_lattice);
}

const D2Q9::LatticeNode *GpuSimulation::get_lattice() {
  cudaMemcpy(host_lattice, device_lattice, sizeof(LatticeNode) * SIZE, cudaMemcpyDeviceToHost);
  return host_lattice;
}