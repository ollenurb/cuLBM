//
// Created by matteo on 3/16/22.
//
#include "Parameters.hpp"

Parameters load_parameters(std::string filename) {
    Parameters configuration;
    INIReader reader(filename);
    if (reader.ParseError() != 0) {
        throw ParseConfigurationException("Error while parsing config file");
    }
    configuration.width = reader.GetInteger(SECTION, "width", 0);
    configuration.height = reader.GetInteger(SECTION, "height", 0);
    configuration.velocity.x = reader.GetReal(SECTION, "velocity.x", 0);
    configuration.velocity.y = reader.GetReal(SECTION, "velocity.y", 0);
    Real viscosity = reader.GetReal(SECTION, "viscosity", .02);
    configuration.omega = 1 / (3 * viscosity + 0.5);
    return configuration;
}
