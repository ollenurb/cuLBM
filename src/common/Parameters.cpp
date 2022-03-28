//
// Created by matteo on 3/16/22.
//
#include "Parameters.hpp"
#include <map>

static std::map<std::string, SimulationType> types_assoc = {
        {STR_REALTIME, REALTIME},
        {STR_PARAVIEW, PARAVIEW},
        {STR_BENCHMARK, BENCHMARK}
};

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
    Real viscosity = reader.GetReal(SECTION, "viscosity", 0);
    configuration.omega = 1 / (3 * viscosity + 0.5);
    configuration.steps = reader.GetInteger(SECTION, "steps", 0);
    configuration.gpu = reader.GetBoolean(SECTION, "gpu", false);
    configuration.grid_size.x = reader.GetInteger(SECTION, "grid_size.x", 1);
    configuration.grid_size.y = reader.GetInteger(SECTION, "grid_size.y", 1);
    /* "parse" ConfigurationType */
    std::string type = reader.Get(SECTION, "type", STR_REALTIME);
    configuration.type = types_assoc[type];
    return configuration;
}
