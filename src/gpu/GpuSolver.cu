//
// Created by matteo on 7/18/21.
//
#include "GpuSolver.cuh"
#include "../common/Exceptions.hpp"
#include <algorithm>

/* Translate a 2D index to a 1D index */
#define index(x, y) x * device::HEIGHT + y

/* Global variables to store block and grid dim */
/* Thread blocks are squares therefore they'll have dimension BLOCK_DIMxBLOCK_DIM */
#define BLOCK_DIM 16
dim3 dim_block;
dim3 dim_grid;

//__device__ void compute_macroscopics(const LatticeNode& node, Vector2D<Real> &new_u, Real &rho);
//
///* ============================================================================================================== */
///* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Device Variables and Constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
///* ============================================================================================================== */
//namespace device {
//  CONSTANT unsigned int WIDTH;
//  CONSTANT unsigned int HEIGHT;
//  CONSTANT Lattice INITIAL_CONFIG;
//  CONSTANT Real W[Q];
//  CONSTANT Vector2D<int> e[Q];
//}
//
///* ============================================================================================================== */
///* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Device-Specific Code and Kernels ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
///* ============================================================================================================== */
//
///* Initialize the device lattice using initial config's values */
//__global__ void init_kernel(LatticeNode *lattice, LatticeNode *lattice_t) {
//  unsigned x_i = blockIdx.x * blockDim.x + threadIdx.x;
//  unsigned y_i = blockIdx.y * blockDim.y + threadIdx.y;
//  unsigned index = index(x_i, y_i);
//  lattice[index] = lattice_t[index] = device::INITIAL_CONFIG; // Initialize both lattices
//
//  /* TODO: TO REMOVE Put a circle at the center of the simulation */
//  unsigned rel_x = device::WIDTH / 2 - x_i;
//  unsigned rel_y = device::HEIGHT / 2 - y_i;
//  double r = sqrt(static_cast<float>(rel_x * rel_x + rel_y * rel_y));
//
//  if(r < min(device::WIDTH, device::HEIGHT) * 0.05) {
//    lattice[index].obstacle = lattice_t[index].obstacle = true;
//    lattice[index].u = lattice_t[index].u = {0, 0};
//    for (int i = 0; i < Q; i++) {
//      lattice[index].f[i] = lattice_t[index].f[i] = 0;
//    }
//  }
//}
//
///* Stream the fluid */
//__global__ void stream(LatticeNode *lattice, LatticeNode *lattice_t) {
//  unsigned int x_i = blockIdx.x * blockDim.x + threadIdx.x;
//  unsigned int y_i = blockIdx.y * blockDim.y + threadIdx.y;
//
//  /* Move the fluid to neighbouring sites */
//  unsigned x_t, y_t;
//  unsigned index, index_t;
//  index = index(x_i, y_i);
//
//  // TODO: Non mi sembra che chiami piu' threads del dovuto, controlla
//  if (index < device::WIDTH * device::HEIGHT) {
//    /* Stream away fluid on each lattice site */
//    for (int i = 0; i < Q; i++) {
//      x_t = x_i + device::e[i].x;
//      y_t = y_i + device::e[i].y;
//      index_t = index(x_t, y_t);
//
//      if (x_t > 0 && y_t > 0 && x_t < device::WIDTH && y_t < device::HEIGHT) {
//        lattice_t[index_t].f[i] = lattice[index].f[i];
//      }
//    }
//
//    /* Handle boundaries */
//    if (y_i == device::HEIGHT - 1 || y_i == 0 || x_i == device::WIDTH - 1 || x_i == 0) {
//      lattice_t[index] = device::INITIAL_CONFIG;
//    }
//  }
//}
//
///* Collide the fluid according to LBM equilibrium density */
//__global__ void collide(LatticeNode *lattice) {
//  unsigned x_i = blockIdx.x * blockDim.x + threadIdx.x;
//  unsigned y_i = blockIdx.y * blockDim.y + threadIdx.y;
//  unsigned index = index(x_i, y_i);
//
//  Real total_density;
//  Real f_eq;
//  Real e_dp_u;
//  Vector2D<Real> new_u{};
//  LatticeNode &cur_node = lattice[index];
//
//  if(!cur_node.obstacle) {
//    compute_macroscopics(cur_node, new_u, total_density);
//
//    /* Compute densities at thermal equilibrium */
//#pragma unroll
//    for (int i = 0; i < Q; i++) {
//      e_dp_u = new_u * device::e[i];
//      f_eq = (total_density * device::W[i]) * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * new_u.mod_sqr()));
//      cur_node.f[i] += D2Q9::OMEGA * (f_eq - cur_node.f[i]);
//    }
//    cur_node.u = new_u;
//  }
//}
//
//__device__ void compute_macroscopics(const LatticeNode& node, Vector2D<Real> &new_u, Real &rho) {
//  // Prepare values
//  rho = 0;
//  new_u = {0, 0};
//
//#pragma unroll
//  for (int i = 0; i < Q; i++) {
//    rho += node.f[i];
//    /* Accumulate the f inside each component of flow_velocity */
//    new_u.x += device::e[i].x * node.f[i]; // U_{x} component
//    new_u.y += device::e[i].y * node.f[i]; // U_{y} component
//  }
//  /* Normalize over Rho */
//  new_u.x = new_u.x / rho;
//  new_u.y = new_u.y / rho;
//}
//
///* Bounce back fluid on obstacles */
//__global__ void bounce(LatticeNode *lattice_t) {
//  unsigned x_i = blockIdx.x * blockDim.x + threadIdx.x;
//  unsigned y_i = blockIdx.y * blockDim.y + threadIdx.y;
//  unsigned index = index(x_i, y_i);
//  LatticeNode &cur_node = lattice_t[index];
//
//  /* Sadly, lots of threads are going to diverge here */
//  if (cur_node.obstacle) {
//    lattice_t[index((x_i + 1), y_i)].f[1] = cur_node.f[3];
//    lattice_t[index(x_i, (y_i + 1))].f[2] = cur_node.f[4];
//    lattice_t[index((x_i - 1), y_i)].f[3] = cur_node.f[1];
//    lattice_t[index(x_i, (y_i - 1))].f[4] = cur_node.f[2];
//    lattice_t[index((x_i + 1), (y_i + 1))].f[5] = cur_node.f[7];
//    lattice_t[index((x_i - 1), (y_i + 1))].f[6] = cur_node.f[8];
//    lattice_t[index((x_i - 1), (y_i - 1))].f[7] = cur_node.f[5];
//    lattice_t[index((x_i + 1), (y_i - 1))].f[8] = cur_node.f[6];
//
//    for(int i = 1; i < Q; i++) {
//      cur_node.f[i] = 0;
//    }
//  }
//}
//
///* ============================================================================================================== */
///* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Class Implementation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
///* ============================================================================================================== */
//GpuSimulation::GpuSimulation(unsigned int w, unsigned int h) : Simulation(w, h) {
//  SIZE = WIDTH * HEIGHT;
////  cudaDeviceProp deviceProp;
//
//  dim_block = dim3(BLOCK_DIM, BLOCK_DIM);
//  dim_grid = dim3((WIDTH + dim_block.x - 1) / dim_block.x, (HEIGHT + dim_block.y - 1) / dim_block.y);
//  /* Allocate space for lattice objects on the device */
//  cudaMemcpyToSymbol(device::WIDTH, &WIDTH, sizeof(unsigned int));
//  cudaMemcpyToSymbol(device::HEIGHT, &HEIGHT, sizeof(unsigned int));
//  cudaMemcpyToSymbol(device::e, &D2Q9::e, sizeof(Vector2D<int>) * Q);
//  cudaMemcpyToSymbol(device::W, &D2Q9::W, sizeof(Real) * Q);
//
//  /* Allocate 2 lattices on the device. The streaming operator needs it */
//  cudaMalloc(&device_lattice, sizeof(LatticeNode) * SIZE);
//  cudaMalloc(&device_lattice_t, sizeof(LatticeNode) * SIZE);
//  host_lattice = new LatticeNode[SIZE];
//
//  /* Compute the initial configuration's parameters */
//  Real e_dp_u;
//  LatticeNode tmp_init_conf;
//  tmp_init_conf.u = D2Q9::VELOCITY;
//  /* Assign each lattice with the equilibrium f */
//#pragma unroll
//  for (int i = 0; i < Q; i++) {
//    e_dp_u = tmp_init_conf.u * D2Q9::e[i];
//    tmp_init_conf.f[i] = D2Q9::W[i] * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * tmp_init_conf.u.mod_sqr()));
//  }
//
//  cudaMemcpyToSymbol(device::INITIAL_CONFIG, &tmp_init_conf, sizeof(LatticeNode));
//  init_kernel<<<dim_grid, dim_block>>>(device_lattice, device_lattice_t);
//  cudaDeviceSynchronize();
//}
//
//GpuSimulation::~GpuSimulation() {
//  cudaFree(device_lattice);
//  cudaFree(device_lattice_t);
//  delete host_lattice;
//}
//
//void GpuSimulation::step() {
//  collide<<<dim_grid, dim_block>>>(device_lattice);
//  stream<<<dim_grid, dim_block>>>(device_lattice, device_lattice_t);
//  bounce<<<dim_grid, dim_block>>>(device_lattice_t);
//  cudaDeviceSynchronize();
//  /* Swap device pointers */
//  std::swap(device_lattice_t, device_lattice);
//}
//
//const D2Q9::LatticeNode *GpuSimulation::get_lattice() {
//  cudaMemcpy(host_lattice, device_lattice, sizeof(LatticeNode) * SIZE, cudaMemcpyDeviceToHost);
//  return host_lattice;
//}

void GpuSolver::step() {

}

GpuSolver::GpuSolver(unsigned w, unsigned h) : Solver(w, h) {
  /* Setup device */
  int device = get_device();
  cudaSetDevice(device);

  /* Initialize host and device data structures */

}

/*
 * Returns the device id that has the maximum compute capability
 */
int GpuSolver::get_device() {
  int device_id;
  int number_of_devices = 0;
  cudaGetDeviceCount(&number_of_devices);
  if (number_of_devices <= 0) {
    throw DeviceNotFoundException("No suitable CUDA device found");
  }

  int highest_compute_capability = 0;
  for (int id = 0; id < number_of_devices; id++) {
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, id);

    /* Check if the current compute capability is the maximum one, and update it accordingly */
    int current_compute_capability = device_properties.major * 100 + device_properties.minor;
    if (current_compute_capability > highest_compute_capability) {
      highest_compute_capability = current_compute_capability;
      device_id = id;
    }
  }

  return device_id;
}

GpuSolver::~GpuSolver() {

}
