//
// Created by matteo on 3/13/22.
//
#ifndef LBM_PARAMETERS_HPP
#define LBM_PARAMETERS_HPP

#include "D2Q9.hpp"
#include "Exceptions.hpp"
#include <config/INIReader.h>
#include <string>

using namespace D2Q9;

#define SECTION "parameters"
#define STR_REALTIME "realtime"
#define STR_PARAVIEW "paraview"
#define STR_BENCHMARK "benchmark"

/* Define the possible types of the simulation */
enum SimulationType {
    BENCHMARK,
    REALTIME,
    PARAVIEW
};

struct Parameters {
    unsigned width, height;
    Vector2D<Real> velocity;
    Real omega;
    SimulationType type;
    unsigned steps;
    bool gpu;
    Vector2D<unsigned> grid_size;
};

Parameters load_parameters(std::string filename);

#endif