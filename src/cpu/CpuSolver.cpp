#include "CpuSolver.hpp"
#include <cmath>

CpuSolver::CpuSolver(Parameters params) : Solver(params) {
    /* Allocate space for the temporary lattice */
    lattice_t = malloc_host(params.width, params.height);

    Real e_dp_u;
    /* Initialize the initial configuration */
    equilibrium_configuration.u = params.velocity;
    /* Assign each lattice with the equilibrium f */
    for (int i = 0; i < Q; i++) {
        e_dp_u = equilibrium_configuration.u * e[i];
        equilibrium_configuration.f[i] = W[i] * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * equilibrium_configuration.u.mod_sqr()));
    }

    /* Assign the simulation lattices with initial configuration's values */
    for (int x = 0; x < params.width; x++) {
        for (int y = 0; y < params.height; y++) {
            /* Initialize flow velocity */
            lattice.u(x, y) = lattice_t.u(x, y) = equilibrium_configuration.u;

            for (int i = 0; i < Q; i++) {
                lattice.f(x, y, i) = lattice_t.f(x, y, i) = equilibrium_configuration.f[i];
            }
        }
    }

    center_circle(obstacle);
}

inline unsigned clamp(unsigned val, unsigned low, unsigned high) {
    return std::min(std::max(val, low), high);
}

/*
 * Stream and bounce inside the same step to speed up computations
  */
void CpuSolver::stream() {
    for (int x = 0; x < params.width; x++) {
        for (int y = 0; y < params.height; y++) {
            if (obstacle(x, y)) {
                bounce(x, y);
            }
        }
    }

    /* "CpuSimulation sites along the edges contain fluid that
   * is always assigned to have the equilibrium number
   * densities for some fixed f and velocity"
   * (Schroeder - CpuSimulation-Boltzmann Fluid Dynamics)
   */
    for (int x = 0; x < params.width; x++) {
        lattice_t.u(x, params.height - 1) = lattice_t.u(x, 0) = equilibrium_configuration.u;
        for (int i = 0; i < Q; i++) {
            lattice_t.f(x, 0, i) = lattice_t.f(x, params.height - 1, i) = equilibrium_configuration.f[i];
        }
    }

    for (int y = 0; y < params.height; y++) {
        lattice_t.u(0, y) = lattice_t.u(params.width - 1, y) = equilibrium_configuration.u;
        for (int i = 0; i < Q; i++) {
            lattice_t.f(0, y, i) = lattice_t.f(params.width - 1, y, i) = equilibrium_configuration.f[i];
        }
    }
}

void CpuSolver::stream(unsigned x, unsigned y) {
    for (int i = 0; i < Q; i++) {
        unsigned x_index, y_index;
        x_index = clamp(x + e[i].x, 0, params.width - 1);
        y_index = clamp(y + e[i].y, 0, params.height - 1);
        lattice_t.f(x_index, y_index, i) = lattice.f(x, y, i);
    }

    /* "Lattice sites along the edges contain fluid that
   * is always assigned to have the equilibrium number
   * densities for some fixed f and velocity"
   * (Schroeder - CpuSimulation-Boltzmann Fluid Dynamics)
   */
    if (y == params.height - 1 || y == 0 || x == params.width - 1 || x == 0) {
        lattice_t.u(x, y) = equilibrium_configuration.u;
        for(int i = 0; i < Q; i++) lattice_t.f(x, y, i) = equilibrium_configuration.f[i];
    }
}

void CpuSolver::collide(unsigned x, unsigned y) {
    if (obstacle(x, y)) return;
    /* Compute the total f of lattice site at position (x, y) */
    Real total_density = 0;
    Vector2D<Real> new_u = {0, 0};

    for (int i = 0; i < Q; i++) {
        Real current_direction = lattice.f(x, y, i);
        total_density += current_direction;
        /* Accumulate the f inside each component of flow_velocity */
        new_u.x += static_cast<Real>(e[i].x) * current_direction;// U_{x} component
        new_u.y += static_cast<Real>(e[i].y) * current_direction;// U_{y} component
    }

    /* Compute average to get the actual value of flow_velocity */
    new_u.x = (total_density > 0) ? (new_u.x / total_density) : 0;
    new_u.y = (total_density > 0) ? (new_u.y / total_density) : 0;

    /* Compute densities at thermal equilibrium */
    /* Equation (8) */
    Real f_eq;
    Real e_dp_u;

    for (int i = 0; i < Q; i++) {
        e_dp_u = new_u * e[i];
        f_eq = (total_density * W[i]) * (1 + (3 * e_dp_u) + (static_cast<Real>(4.5) * (e_dp_u * e_dp_u)) -
                                         (static_cast<Real>(1.5) * new_u.mod_sqr()));
        lattice.f(x, y, i) += params.omega * (f_eq - lattice.f(x, y, i));
    }
    lattice.u(x, y) = new_u;
}

// IMPROVEMENT: You can store indexes on two separate arrays, so that you can later apply changes in a for loop
void CpuSolver::bounce(unsigned x, unsigned y) {
    if(!obstacle(x, y)) return;
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

void CpuSolver::step() {
    for(unsigned x = 0; x < params.width; x++) {
       for(unsigned y = 0; y < params.height; y++) {
            collide(x, y);
            stream(x, y);
            bounce(x, y);
       }
    }
    std::swap(lattice, lattice_t);
}

Lattice<Host> &CpuSolver::get_lattice() { return lattice; }