//
// Created by matteo on 7/20/21.
//

#pragma once

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT
#endif
/* This is needed to differentiate between host and device code at compile time */
#ifdef __CUDA_ARCH__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif
