//
// Created by matteo on 7/18/21.
//
#include "../common/Exceptions.hpp"
#include "GpuSolver.cuh"
#include "GpuUtils.cuh"
#include <algorithm>

/* Global variables to store block and grid dim */
/* Thread blocks are squares therefore they'll have dimension BLOCK_DIMxBLOCK_DIM */
#define BLOCK_DIM 16
dim3 dim_block;
dim3 dim_grid;

/* +=============================================+ */
/* +=============+ Device Specific +=============+ */
/* +=============================================+ */
namespace device {
    CONSTANT Parameters params;
    CONSTANT LatticeNode equilibrium_config;
    CONSTANT Vector2D<int> e[Q];
    CONSTANT Real W[Q];
    CONSTANT dim3 task_size;
}// namespace device

__global__ void init_kernel(Lattice<Device> lattice, Lattice<Device> lattice_t) {
    unsigned x_from = device::task_size.x * (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned y_from = device::task_size.y * (blockIdx.y * blockDim.y + threadIdx.y);
    unsigned x_to = x_from + device::task_size.x;
    unsigned y_to = y_from + device::task_size.y;

    for (unsigned x_i = x_from; x_i < x_to; x_i++) {
        for (unsigned y_i = y_from; y_i < y_to; y_i++) {
            /* Initialize lattices with the initial configuration */
            lattice_t.u(x_i, y_i) = lattice.u(x_i, y_i) = device::equilibrium_config.u;

#pragma unroll
            for (int i = 0; i < Q; i++) {
                lattice.f(x_i, y_i, i) = lattice_t.f(x_i, y_i, i) = device::equilibrium_config.f[i];
            }
        }
    }
}

__global__ void stream_kernel(Lattice<Device> lattice, Lattice<Device> lattice_t) {
    unsigned x_from = device::task_size.x * (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned y_from = device::task_size.y * (blockIdx.y * blockDim.y + threadIdx.y);
    unsigned x_to = x_from + device::task_size.x;
    unsigned y_to = y_from + device::task_size.y;

    for (unsigned x_i = x_from; x_i < x_to; x_i++) {
        for (unsigned y_i = y_from; y_i < y_to; y_i++) {
            /* Move the fluid to neighbouring sites */
            unsigned x_t, y_t;

            /* Stream away fluid on each lattice site */
#pragma unroll
            for (int i = 0; i < Q; i++) {
                x_t = x_i + device::e[i].x;
                y_t = y_i + device::e[i].y;

                if (x_t > 0 && y_t > 0 && x_t < device::params.width && y_t < device::params.height) {
                    lattice_t.f(x_t, y_t, i) = lattice.f(x_i, y_i, i);
                }
            }

            /* Handle boundaries */
            if (y_i == device::params.height - 1 || y_i == 0 || x_i == device::params.width - 1 || x_i == 0) {
                lattice_t.u(x_i, y_i) = device::equilibrium_config.u;
                for (int i = 0; i < Q; i++) lattice_t.f(x_i, y_i, i) = device::equilibrium_config.f[i];
            }
        }
    }
}

__global__ void collide_kernel(Lattice<Device> lattice, Bitmap<Device> obstacle) {
    unsigned x_from = device::task_size.x * (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned y_from = device::task_size.y * (blockIdx.y * blockDim.y + threadIdx.y);
    unsigned x_to = x_from + device::task_size.x;
    unsigned y_to = y_from + device::task_size.y;

    for (unsigned x_i = x_from; x_i < x_to; x_i++) {
        for (unsigned y_i = y_from; y_i < y_to; y_i++) {

            Real rho;
            Real f_eq;
            Real e_dp_u;
            Vector2D<Real> new_u{};

            if (!obstacle(x_i, y_i)) {
                // Prepare values
                rho = 0;
                new_u = {0, 0};

#pragma unroll
                for (int i = 0; i < Q; i++) {
                    Real val = lattice.f(x_i, y_i, i);
                    rho += val;
                    /* Accumulate the f inside each component of flow_velocity */
                    new_u.x += device::e[i].x * val;// U_{x} component
                    new_u.y += device::e[i].y * val;// U_{y} component
                }
                /* Normalize over Rho */
                new_u.x = new_u.x / rho;
                new_u.y = new_u.y / rho;

                /* Compute densities at thermal equilibrium */
#pragma unroll
                for (int i = 0; i < Q; i++) {
                    e_dp_u = new_u * device::e[i];
                    f_eq = (rho * device::W[i]) * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * new_u.mod_sqr()));
                    lattice.f(x_i, y_i, i) += device::params.omega * (f_eq - lattice.f(x_i, y_i, i));
                }
                lattice.u(x_i, y_i) = new_u;
            }
        }
    }
}

__global__ void bounce_kernel(Lattice<Device> lattice_t, Bitmap<Device> obstacle) {
    unsigned x_from = device::task_size.x * (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned y_from = device::task_size.y * (blockIdx.y * blockDim.y + threadIdx.y);
    unsigned x_to = x_from + device::task_size.x;
    unsigned y_to = y_from + device::task_size.y;

    for (unsigned x_i = x_from; x_i < x_to; x_i++) {
        for (unsigned y_i = y_from; y_i < y_to; y_i++) {
            /* Sadly, lots of threads are going to diverge here */
            if (obstacle(x_i, y_i)) {
                lattice_t.f((x_i + 1), y_i, 1) = lattice_t.f(x_i, y_i, 3);
                lattice_t.f(x_i, (y_i + 1), 2) = lattice_t.f(x_i, y_i, 4);
                lattice_t.f((x_i - 1), y_i, 3) = lattice_t.f(x_i, y_i, 1);
                lattice_t.f(x_i, (y_i - 1), 4) = lattice_t.f(x_i, y_i, 2);
                lattice_t.f((x_i + 1), (y_i + 1), 5) = lattice_t.f(x_i, y_i, 7);
                lattice_t.f((x_i - 1), (y_i + 1), 6) = lattice_t.f(x_i, y_i, 8);
                lattice_t.f((x_i - 1), (y_i - 1), 7) = lattice_t.f(x_i, y_i, 5);
                lattice_t.f((x_i + 1), (y_i - 1), 8) = lattice_t.f(x_i, y_i, 6);

                for (int i = 1; i < Q; i++) {
                    lattice_t.f(x_i, y_i, i) = 0;
                }
            }
        }
    }
}

/* +=============================================+ */
/* +===========+ Class Implementation +==========+ */
/* +=============================================+ */
void GpuSolver::step() {
    collide_kernel<<<dim_grid, dim_block>>>(device_lattice, device_obstacle);
    stream_kernel<<<dim_grid, dim_block>>>(device_lattice, device_lattice_t);
    bounce_kernel<<<dim_grid, dim_block>>>(device_lattice_t, device_obstacle);
    cudaDeviceSynchronize();
    /* Swap device pointers */
    std::swap(device_lattice_t, device_lattice);
}

GpuSolver::GpuSolver(Parameters params) : Solver(params) {
    /* Setup device */
    int device = get_device();
    cudaSetDevice(device);

    /* Initialize host and device data structures */
    device_lattice = malloc_device(params.width, params.height);
    device_lattice_t = malloc_device(params.width, params.height);
    device_obstacle = malloc_device<bool>(params.width, params.height);

    cudaMemcpyToSymbol(device::e, &D2Q9::e, sizeof(Vector2D<int>) * Q);
    cudaMemcpyToSymbol(device::W, &D2Q9::W, sizeof(Real) * Q);
    cudaMemcpyToSymbol(device::params, &params, sizeof(Parameters));

    /* Compute grid and block size */
    dim_block = dim3(BLOCK_DIM, BLOCK_DIM);
    dim_grid = dim3(params.grid_size.x, params.grid_size.y);

    /* Compute task size */
    dim3 task_size = dim3(params.width / (dim_block.x * dim_grid.x), params.height / (dim_block.y * dim_grid.y));
    cudaMemcpyToSymbol(device::task_size, &task_size, sizeof(dim3));

    printf("Task Size: %d x %d\n", task_size.x, params.height / task_size.y);
    printf("GridSize: %d x %d\n", dim_grid.x, dim_grid.y);

    /* TODO: Refactor into compute_equilibrium(...);*/
    /* Compute equilibrium f for the initial configuration */
    Real e_dp_u;
    LatticeNode tmp_init_conf;
    tmp_init_conf.u = params.velocity;
    /* Assign each lattice with the equilibrium f */
#pragma unroll
    for (int i = 0; i < Q; i++) {
        e_dp_u = tmp_init_conf.u * D2Q9::e[i];
        tmp_init_conf.f[i] = D2Q9::W[i] * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * tmp_init_conf.u.mod_sqr()));
    }

    /* Put a circle at the center, then synchronize with device */
    center_circle(obstacle);
    host_to_device(obstacle, device_obstacle);

    cudaMemcpyToSymbol(device::equilibrium_config, &tmp_init_conf, sizeof(LatticeNode));
    init_kernel<<<dim_grid, dim_block>>>(device_lattice, device_lattice_t);
    cudaDeviceSynchronize();
    device_to_host(device_lattice.f, lattice.f);
    device_to_host(device_lattice.u, lattice.u);
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
    free_lattice(device_lattice);
    free_lattice(device_lattice_t);
    free_array(device_obstacle);
    cudaDeviceReset();
}

Lattice<Host> &GpuSolver::get_lattice() {
    /* Synchronize with device lattice */
    device_to_host(device_lattice.f, lattice.f);
    device_to_host(device_lattice.u, lattice.u);
    return lattice;
}
