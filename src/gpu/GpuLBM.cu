//
// Created by matteo on 7/18/21.
//
#include "GpuLBM.cuh"
#include "GpuVector2D.cuh"
#include "../common/Utils.hpp"

#define LOG_X 11
#define LOG_Y 11

struct LatticeNode_t {
    double density[Q] = WEIGHTS;
    double total_density = 1.0;
    GpuVector2D<double> macroscopic_velocity = {0, 0};
};

__constant__ unsigned int D_WIDTH;
__constant__ unsigned int D_HEIGHT;
__constant__ LatticeNode D_INITIAL_CONFIG;
__constant__ const double D_W[Q] = WEIGHTS;
__constant__ GpuVector2D<int> D_e[Q] =
        {
                { 0, 0}, { 1,  0}, {0,  1},
                {-1, 0}, { 0, -1}, {1,  1},
                {-1, 1}, {-1, -1}, {1, -1}
        };

const double W[Q] = WEIGHTS;
constexpr static const double VISCOSITY = 0.020;
constexpr static const double VELOCITY = 0.070;
constexpr static const double OMEGA = 1 / (3 * VISCOSITY + 0.5);


unsigned BLOCK_DIM = 20;
dim3 dim_block;
dim3 dim_grid;

__device__ void bounce()
{

}

__device__ inline unsigned clamp(unsigned val, unsigned low, unsigned high)
{
    return min(max(val, low), high);
}

__device__ void stream(LatticeNode *lattice, LatticeNode *lattice_t)
{
    int x_i = blockIdx.x * blockDim.x + threadIdx.x;
    int y_i = blockIdx.y * blockDim.y + threadIdx.y;

    /* Move the fluid to neighbouring sites */
    unsigned x_index, y_index;
    unsigned index = x_i * D_HEIGHT + y_i;
    unsigned myIndex = index;

    /* "LBM sites along the edges contain fluid that
     * is always assigned to have the equilibrium number
     * densities for some fixed density and velocity"
     * (Schroeder - LBM-Boltzmann Fluid Dynamics)
     */
    if(y_i != D_HEIGHT-1 && y_i != 0 && x_i != D_WIDTH-1 && x_i != 0) {
        for(int i = 0; i < Q; i++) {
            x_index = x_i + D_e[i].x;
            y_index = y_i + D_e[i].y;
            index = x_index * D_HEIGHT + y_index;
            lattice_t[index].density[i] = lattice[myIndex].density[i];
        }
    }
    __threadfence();
    if(y_i == D_HEIGHT-1 || y_i == 0 || x_i == D_WIDTH-1 || x_i == 0) {
        lattice_t[x_i * D_HEIGHT + y_i] = D_INITIAL_CONFIG;
    }
}

__device__ void collide(LatticeNode* lattice)
{
    unsigned x_i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y_i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned index = x_i * D_HEIGHT + y_i;
    if(index > D_WIDTH * D_HEIGHT) {
        printf("ERROR! Index out of bounds");
    }

    double total_density;
    double density_eq;
    double e_dp_u;
    GpuVector2D<double> u{};

    LatticeNode &cur_node = lattice[index];
    /* Compute the total density of lattice site at position (x, y) */
    total_density = 0.0;
    u.x = u.y = 0.0;

    for(int i = 0; i < Q; i++) {
        total_density += cur_node.density[i];
        /* Accumulate the density inside each component of flow_velocity */
        u.x += D_e[i].x * cur_node.density[i]; // U_{x} component
        u.y += D_e[i].y * cur_node.density[i]; // U_{y} component
    }

    /* Compute average to get the actual value of flow_velocity */
    /* "Cast" to 0 if the velocity is negative */
    u.x = max(0.0, u.x / total_density);
    u.y = max(0.0, u.y / total_density);

    /* Compute densities at thermal equilibrium */
    /* Equation (8) */
    for(int i = 0; i < Q; i++) {
        e_dp_u = u * D_e[i];
        density_eq = total_density * D_W[i] * (1 + (3*e_dp_u) + (4.5*(e_dp_u * e_dp_u)) - (1.5 * u.mod_sqr()));
        cur_node.density[i] += OMEGA * (density_eq - cur_node.density[i]);
    }
    cur_node.total_density = total_density;
    cur_node.macroscopic_velocity = u;
}

/* Initialize the device lattice using initial config's values */
__global__ void init_kernel(LatticeNode* lattice, LatticeNode* lattice_t)
{
    unsigned x_i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y_i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned index = x_i * D_HEIGHT + y_i;
    lattice[index] = lattice_t[index] = D_INITIAL_CONFIG;

    /* Put a spike of density at the center */
    if(x_i > 10 && x_i < 20 && y_i > 10 && y_i < 20) {
        for(double & i : lattice[index].density)
            i += 0.02;
    }
}

__global__ void step_kernel(LatticeNode* lattice, LatticeNode* lattice_t)
{
    collide(lattice);
    __threadfence();
    stream(lattice, lattice_t);
    bounce();
}

GpuLBM::GpuLBM(unsigned int w, unsigned int h) : Simulation(w, h) {
    GpuVector2D<int> e[Q] =
            {
                    { 0, 0}, { 1,  0}, {0,  1},
                    {-1, 0}, { 0, -1}, {1,  1},
                    {-1, 1}, {-1, -1}, {1, -1}
            };
    SIZE = WIDTH * HEIGHT;
    dim_block = dim3(BLOCK_DIM, BLOCK_DIM);
    dim_grid = dim3(WIDTH / dim_block.x, HEIGHT / dim_block.y);
    /* Allocate space for lattice objects on the device */
    cudaMemcpyToSymbol(D_WIDTH, &WIDTH, sizeof(unsigned int));
    cudaMemcpyToSymbol(D_HEIGHT, &HEIGHT, sizeof(unsigned int));

    cudaMalloc(&device_lattice, sizeof(LatticeNode) * SIZE);
    cudaMalloc(&device_lattice_t, sizeof(LatticeNode) * SIZE);

    host_lattice = new LatticeNode[SIZE];

    /* Initialize the initial configuration */
    double e_dp_u;
    LatticeNode host_initial_config;
    host_initial_config.macroscopic_velocity.x = VELOCITY;
    host_initial_config.macroscopic_velocity.y = 0;
    /* Assign each lattice with the equilibrium density */
    for(int i = 0; i < Q; i++) {
        e_dp_u = host_initial_config.macroscopic_velocity * e[i];
        host_initial_config.density[i] = W[i] * (1 + (3*e_dp_u) + (4.5*(e_dp_u * e_dp_u)) - (1.5 * host_initial_config.macroscopic_velocity.mod_sqr()));
    }

    cudaMemcpyToSymbol(D_INITIAL_CONFIG, &host_initial_config, sizeof(LatticeNode));
    /* Initialize */
    printf("Block: %d, %d, %d, Grid: %d, %d, %d\n", dim_block.x, dim_block.y, dim_block.z, dim_grid.x, dim_grid.y, dim_grid.z);
    init_kernel<<<dim_grid, dim_block>>>(device_lattice, device_lattice_t);
    cudaDeviceSynchronize();
}

GpuLBM::~GpuLBM() {
    cudaFree(device_lattice);
    cudaFree(device_lattice_t);
    delete host_lattice;
}

void GpuLBM::render(SDL_Texture *screen) {
    cudaMemcpy(host_lattice, device_lattice, sizeof(LatticeNode) * SIZE, cudaMemcpyDeviceToHost);
    void *pixels;
    int pitch;
    Uint32 *dest;
    float b;

    if (SDL_LockTexture(screen, nullptr, &pixels, &pitch) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't lock texture: %s\n", SDL_GetError());
    }

    for(int y = 0; y < HEIGHT; y++) {
        dest = (Uint32*)((Uint8*) pixels + y * pitch);
        for(int x = 0; x < WIDTH; x++) {
            b = std::min(host_lattice[x * HEIGHT + y].macroscopic_velocity.modulus() * 4, 1.0);
            *(dest + x) = utils::HSBtoRGB(0.5, 1.0, b);
        }
    }
    SDL_UnlockTexture(screen);
}

void GpuLBM::step()
{
    cudaDeviceSynchronize();
    step_kernel<<<dim_grid, dim_block>>>(device_lattice, device_lattice_t);
    LatticeNode *tmp = device_lattice_t;
    device_lattice_t = device_lattice;
    device_lattice = tmp;
}
