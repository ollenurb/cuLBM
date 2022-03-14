//
// Created by matteo on 3/3/22.
//
#pragma once
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

class UnsupportedDeviceTypeException : std::runtime_error {
public:
    UnsupportedDeviceTypeException(std::string const &what) : std::runtime_error(what.c_str()) {}
};
