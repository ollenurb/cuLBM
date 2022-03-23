#ifndef LBM_CPU_SOLVER_HPP
#define LBM_CPU_SOLVER_HPP

#include "../common/Solver.hpp"

using namespace D2Q9;

class CpuSolver : public Solver {
private:
    Lattice<Host> lattice_t;

    void stream();

    void collide();


    void stream(unsigned x, unsigned y);

    void collide(unsigned x, unsigned y);

    void bounce(unsigned x, unsigned y);

public:
    CpuSolver(Parameters params);
    ~CpuSolver() { free_lattice(lattice_t); }

    virtual Lattice<Host>& get_lattice() override;

    void step() override;
};

#endif