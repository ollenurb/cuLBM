#include "common/Engine.hpp"
#include <chrono>
#include <iostream>
/* If the GPU compilation flag is enabled, then include the GPU-Specific version */
#ifdef GPU_ENABLED
#include "gpu/GpuLBM.cuh"
#else
#include "cpu/LBM.hpp"
#endif

#define WIDTH 10
#define HEIGHT 10

//void run_benchmark(unsigned long steps)
//{
//    using std::chrono::high_resolution_clock;
//    using std::chrono::duration_cast;
//    using std::chrono::duration;
//    using std::chrono::milliseconds;
//
//    std::cout << "Simulating " << steps << " time steps of a "
//              << WIDTH << "x" << HEIGHT << " host_lattice"
//              << std::endl;
//
//    LBM host_lattice(WIDTH, HEIGHT);
//    auto t0 = high_resolution_clock::now();
//    while(steps > 0) {
//        host_lattice.step();
//        steps--;
//    }
//    auto t1 = high_resolution_clock::now();
//
//    /* Getting number of milliseconds as an integer */
//    auto ms_int = duration_cast<milliseconds>(t1 - t0);
//
//    std::cout << "The program took " << ms_int.count() << "ms to complete" << std::endl;
//}

int main(int argc, char** argv)
{

#ifdef GPU_ENABLED
    GpuLBM lattice(WIDTH, HEIGHT);
#else
    LBM host_lattice(WIDTH, HEIGHT);
#endif

    lattice.step();

//    Engine engine(host_lattice);
//    engine.run();
}
