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

struct Parameters {
    unsigned width, height;
    Vector2D<Real> velocity;
    Real omega;
};

Parameters load_parameters(std::string filename);

#endif