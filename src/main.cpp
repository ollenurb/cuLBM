#include "common/Engine.hpp"
#include <chrono>
#include <iostream>
/* If the GPU compilation flag is enabled, then include the GPU-Specific version */
#ifdef GPU_ENABLED

#include "gpu/GpuSimulation.cuh"

#else
#include "cpu/CpuSimulation.hpp"
#endif

#define WIDTH 1600
#define HEIGHT 1040

void run_benchmark(unsigned long steps) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
#ifdef GPU_ENABLED
    std::cout << "+=======================+ Benchmarking GPU accelerated version +========================+ "
              << std::endl;
#else
    std::cout << "+==============================+ Benchmarking CPU version +=============================+ " << std::endl;
#endif
    std::cout << "Simulating " << steps << " time steps of a "
              << WIDTH << "x" << HEIGHT << " host_lattice"
              << std::endl;


#ifdef GPU_ENABLED
    GpuSimulation lattice(WIDTH, HEIGHT);
#else
    LBM lattice(WIDTH, HEIGHT);
#endif

    auto t0 = high_resolution_clock::now();
    while (steps > 0) {
        lattice.step();
        steps--;
    }
    auto t1 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer */
    auto ms_int = duration_cast<milliseconds>(t1 - t0);

    std::cout << "The program took " << ms_int.count() << "ms to complete" << std::endl;
}

int main(int argc, char **argv) {

#ifdef GPU_ENABLED
    GpuSimulation lattice(WIDTH, HEIGHT);
#else
    LBM lattice(WIDTH, HEIGHT);
#endif

//    lattice.step();

    Engine engine(lattice);
    engine.run();
//    run_benchmark(1000);
}
