#pragma once

#include "Configuration.hpp"
#include "D2Q9.hpp"
#include "Array.hpp"

class Solver {
protected:
    Lattice<Host> lattice;
    Array<bool, Host> obstacle;

public:
    Configuration config;

    Solver(Configuration config) : config(config) { }

    virtual ~Solver() {
        lattice.free();
        obstacle.free();
    }

    virtual void step() = 0;

    virtual Lattice<Host> *get_lattice() = 0;
};