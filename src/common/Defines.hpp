//
// Created by matteo on 7/20/21.
//
#ifndef LBM_DEFINES_HPP
#define LBM_DEFINES_HPP

// Select precision
typedef float Real;

/* This is needed to differentiate between host and device code at compile time */
#ifdef __CUDA_ARCH__
  #define CONSTANT __constant__
  #define HOST_DEVICE __host__ __device__
  #define DEVICE __device__
  #define HOST __host__
#else
  #define HOST_DEVICE
  #define DEVICE
  #define HOST
  #define CONSTANT
#endif

/* Define allocation types */
struct Host { };
struct Device { };

#endif