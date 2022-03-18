//
// Created by matteo on 3/3/22.
//
#ifndef LBM_EXCEPTIONS_HPP
#define LBM_EXCEPTIONS_HPP

#include <exception>
#include <stdexcept>

class DeviceNotFoundException : std::runtime_error {
public:
    DeviceNotFoundException(std::string const &what) : std::runtime_error(what.c_str()) {}
};

class ParseConfigurationException : std::runtime_error {
public:
    ParseConfigurationException(std::string const &what) : std::runtime_error(what.c_str()) {}
};

#endif