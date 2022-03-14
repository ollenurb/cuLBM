//
// Created by matteo on 3/13/22.
//
#pragma once

#include "D2Q9.hpp"
#include "Exceptions.hpp"
#include <config/INIReader.h>
#include <image/EasyBMP.hpp>
#include <string>

using namespace D2Q9;

#define SECTION "simulation"

struct Configuration {
    unsigned width, height;
    Vector2D<Real> velocity;
    Real omega;

    HOST void load(std::string filename) {
        INIReader reader(filename);
        if (reader.ParseError() != 0) {
            throw ParseConfigurationException("Error while parsing config file");
        }
        width = reader.GetInteger(SECTION, "width", 0);
        height = reader.GetInteger(SECTION, "height", 0);
        velocity.x = reader.GetReal(SECTION, "velocity.x", 0);
        velocity.y = reader.GetReal(SECTION, "velocity.y", 0);
        Real viscosity = reader.GetReal(SECTION, "viscosity", .02);
        omega = 1 / (3 * viscosity + 0.5);
    }


    HOST_DEVICE inline unsigned index(unsigned x, unsigned y) {
        return  x * height + y;
    }

    HOST_DEVICE inline unsigned index(unsigned x, unsigned y, unsigned z) {
        return x * height + y;
    }
};
