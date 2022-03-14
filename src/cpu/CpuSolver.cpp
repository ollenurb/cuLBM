#include "CpuSolver.hpp"
#include <algorithm>
#include <cmath>

CpuSolver::CpuSolver(Configuration config) : Solver(config) {
    /* Allocate space */
    lattice.init(config.width, config.height);
    lattice_t.init(config.width, config.height);
    obstacle.init(config.width * config.height);

    Real e_dp_u;
    /* Initialize the initial configuration */
    initial_config.u = VELOCITY;
    /* Assign each lattice with the equilibrium f */
    for (int i = 0; i < Q; i++) {
        e_dp_u = initial_config.u * e[i];
        initial_config.f[i] = W[i] * (1 + (3 * e_dp_u) + (4.5f * (e_dp_u * e_dp_u)) - (1.5f * initial_config.u.mod_sqr()));
    }

    /* Assign the simulation lattices with initial configuration's values */
    for (int x = 0; x < config.width; x++) {
        for (int y = 0; y < config.height; y++) {
            /* Initialize flow velocity */
            unsigned index = config.index(x, y);
            lattice.u[index] = lattice_t.u[index] = initial_config.u;

            for (int i = 0; i < Q; i++) {
                unsigned k = config.index(x, y, i);
                lattice.f[k] = lattice_t.f[k] = initial_config.f[i];
            }

            /* TODO: To remove, its just to put a circle in the center */
            unsigned rel_x = config.width / 2 - x;
            unsigned rel_y = config.height / 2 - y;
            double r = sqrt(rel_x * rel_x + rel_y * rel_y);
            if (r < std::min(config.width, config.height) * 0.2) {
                obstacle[index] = true;
                lattice.u[index] = lattice_t.u[index] = {0, 0};
                for (int i = 0; i < Q; i++) {
                    unsigned k = config.index(x, y, i);
                    lattice.f[k] = lattice_t.f[k] = 0;
                }
            }
        }
    }
}

inline unsigned clamp(unsigned val, unsigned low, unsigned high) {
    return std::min(std::max(val, low), high);
}

void CpuSolver::stream() {
    /* Move the fluid to neighbouring sites */
    unsigned x_index, y_index;

    for (int x = 0; x < config.width; x++) {
        for (int y = 0; y < config.height; y++) {
            for (int i = 0; i < Q; i++) {
                x_index = clamp(x + e[i].x, 0, config.width - 1);
                y_index = clamp(y + e[i].y, 0, config.height - 1);
                unsigned index_t = config.index(x_index, y_index, i);
                unsigned index = config.index(x, y, i);
                lattice_t.f[index_t] = lattice.f[index];
            }
        }
    }

    /* "CpuSimulation sites along the edges contain fluid that
   * is always assigned to have the equilibrium number
   * densities for some fixed f and velocity"
   * (Schroeder - CpuSimulation-Boltzmann Fluid Dynamics)
   */
    // TODO: Can be changed
    for (int x = 0; x < config.width; x++) {
        lattice_t.u[index(x, config.height - 1)] = lattice_t.u[index(x, 0)] = initial_config.u;
        for (int i = 0; i < Q; i++) {
            lattice_t.f[index(x, 0, i)] = lattice_t.f[index(x, config.height - 1, i)] = initial_config.f[i];
        }
    }

    for (int y = 0; y < config.height; y++) {
        lattice_t.u[index(0, y)] = lattice_t.u[index(config.width - 1, y)] = initial_config.u;
        for (int i = 0; i < Q; i++) {
            lattice_t.f[index(0, y, i)] = lattice_t.f[index(config.width - 1, y, i)] = initial_config.f[i];
        }
    }
}

void CpuSolver::collide() {
    for (int x = 0; x < config.width; x++) {
        for (int y = 0; y < config.height; y++) {
            unsigned idx = index(x, y);
            if (obstacle[idx]) continue;
            /* Compute the total f of lattice site at position (x, y) */
            Real total_density = 0;
            Vector2D<Real> new_u = {0, 0};

            for (int i = 0; i < Q; i++) {
                Real current_direction = lattice.f[index(x, y, i)];
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
                unsigned k = index(x, y, i);
                e_dp_u = new_u * e[i];
                f_eq = (total_density * W[i]) * (1 + (3 * e_dp_u) + (static_cast<Real>(4.5) * (e_dp_u * e_dp_u)) -
                                                 (static_cast<Real>(1.5) * new_u.mod_sqr()));
                lattice.f[k] += OMEGA * (f_eq - lattice.f[k]);
            }
            lattice.u[idx] = new_u;
        }
    }
}

// IMPROVEMENT: You can store indexes on two separate arrays, so that you can later apply changes in a for loop
void CpuSolver::bounce() {
    for (int x = 1; x < config.width; x++) {
        for (int y = 1; y < config.height; y++) {
            if (obstacle[index(x, y)]) {
                lattice_t.f[index(x + 1, y, 1)] = lattice_t.f[index(x, y, 3)];
                lattice_t.f[index(x, y + 1, 2)] = lattice_t.f[index(x, y, 4)];
                lattice_t.f[index(x - 1, y, 3)] = lattice_t.f[index(x, y, 1)];
                lattice_t.f[index(x, y - 1, 4)] = lattice_t.f[index(x, y, 2)];
                lattice_t.f[index(x + 1, y + 1, 5)] = lattice_t.f[index(x, y, 7)];
                lattice_t.f[index(x - 1, y + 1, 6)] = lattice_t.f[index(x, y, 8)];
                lattice_t.f[index(x - 1, y - 1, 7)] = lattice_t.f[index(x, y, 5)];
                lattice_t.f[index(x + 1, y - 1, 8)] = lattice_t.f[index(x, y, 6)];

                for (int i = 1; i < Q; i++) {
                    lattice_t.f[index(x, y, i)] = 0;
                }
            }
        }
    }
}

void CpuSolver::step() {
    collide();
    stream();
    bounce();
    std::swap(lattice, lattice_t);
}

Lattice<Host> *CpuSolver::get_lattice() {
    return &lattice;
}
