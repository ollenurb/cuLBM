//
// Created by matteo on 7/18/21.
//

#include "GpuLBM.cuh"

__global__ void init_kernel(GpuMatrix<LatticeNode>* lattice, LatticeNode* initial_config)
{
    unsigned x_i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y_i = blockIdx.y * blockDim.y + threadIdx.y;
    (*lattice)(x_i, y_i) = (*initial_config);
}

__global__ void step_kernel(GpuMatrix<LatticeNode>* lattice, GpuMatrix<LatticeNode>* lattice_t)
{

}


GpuLBM::GpuLBM(unsigned int w, unsigned int h) : Simulation(w, h), host_lattice(w, h), host_lattice_t(w, h) {
    /* Allocate space for lattice objects on the device */
    cudaMalloc(&device_lattice, sizeof(GpuMatrix<LatticeNode>));
    cudaMalloc(&device_lattice_t, sizeof(GpuMatrix<LatticeNode>));
    cudaMemcpy(device_lattice, &host_lattice, sizeof(GpuMatrix<LatticeNode>), cudaMemcpyHostToDevice);
    cudaMemcpy(device_lattice_t, &host_lattice_t, sizeof(GpuMatrix<LatticeNode>), cudaMemcpyHostToDevice);

    double e_dp_u;
    /* Initialize the initial configuration */
    host_initial_config.macroscopic_velocity.x = VELOCITY;
    host_initial_config.macroscopic_velocity.y = 0;
    /* Assign each lattice with the equilibrium density */
    for(int i = 0; i < Q; i++) {
        e_dp_u = e[i] * host_initial_config.macroscopic_velocity;
        host_initial_config.density[i] = W[i] * (1 + (3*e_dp_u) + (4.5*(e_dp_u * e_dp_u)) - (1.5 * host_initial_config.macroscopic_velocity.mod_sqr()));
    }

    /* Allocate space for the initial configuration on the device */
    cudaMalloc(&device_initial_config, sizeof(LatticeNode));
    cudaMemcpy(device_initial_config, &host_initial_config, sizeof(LatticeNode), cudaMemcpyHostToDevice);
}

GpuLBM::~GpuLBM() {
    cudaFree(device_lattice);
    cudaFree(device_lattice_t);
    cudaFree(device_initial_config);
}

void GpuLBM::render(SDL_Texture *) {

}

void GpuLBM::step() {
    unsigned BLOCK_DIM = 10;
    dim3 dim_block(BLOCK_DIM, BLOCK_DIM);
    dim3 dim_grid(WIDTH / dim_block.x, HEIGHT / dim_block.y);
    init_kernel<<<dim_grid, dim_block>>>(device_lattice, device_initial_config);
    cudaDeviceSynchronize();

    for(int i = 0; i < HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
           printf("%.4f\n", host_lattice(i, j).total_density);
        }
    }
    cudaFree(device_lattice);
}

__device__ void bounce()
{

}

__device__ void stream()
{

}

__device__ void collide()
{

}





