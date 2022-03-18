#ifndef LBM_SOLVER_HPP
#define LBM_SOLVER_HPP

#include "D2Q9.hpp"
#include "Lattice.hpp"
#include "Parameters.hpp"
#include "Bitmap.hpp"

class Solver {
protected:
    Lattice<Host> lattice;
    Bitmap<Host> obstacle;
    LatticeNode equilibrium_configuration;

public:
    Parameters params;

    /* Initialize lattice and obstacle */
    Solver(Parameters params) : params(params) {
        obstacle = malloc_host<bool>(params.width, params.height);
        lattice = malloc_host(params.width, params.height);
    }

    /* Free previously allocated space */
    virtual ~Solver() {
        free_array(obstacle);
        free_lattice(lattice);
    }

    virtual void step() = 0;

    /* GetLattice is virtual since in the GPU version we need to first synchronize the array */
    virtual Lattice<Host> &get_lattice() = 0;
    Bitmap<Host> &get_obstacle() { return obstacle; };
};

#endif