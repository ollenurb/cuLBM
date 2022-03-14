#pragma once

#include "../common/Solver.hpp"

using namespace D2Q9;

class CpuSolver : public Solver {
private:
    Lattice<Host> lattice_t;
    LatticeNode initial_config;

    inline unsigned index(unsigned x, unsigned y) { return x + y; }
    inline unsigned index(unsigned x, unsigned y, unsigned z) { return x + y + z; }

    void stream();

    void collide();

    void bounce();

public:
    CpuSolver(Configuration config);

    virtual Lattice<Host> *get_lattice() override;

    void step() override;
};