//
// Created by matteo on 9/16/21.
//

#pragma once

#include "../D2Q9.hpp"
#include "../Solver.hpp"

class VtkEngine {
private:
    Configuration config;
    const unsigned int STEPS;
    Solver &simulation;

    // VTK-related functions
    void write_header(FILE *) const;
    void write_data(FILE *file, Lattice<Host> *lattice);

public:
    explicit VtkEngine(Solver &, unsigned int);
    ~VtkEngine();
    void run();
};