#include "common/engines/SdlEngine.hpp"
#include "cpu/CpuSolver.hpp"
#include "gpu/GpuSolver.cuh"
#include <chrono>
#include <iostream>

/*
 *  Run the solver for a given number of steps
 *  Returns the number of Lattice Updates Per Seconds
 */
float run_benchmark(Solver &solver, Parameters params) {
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    unsigned long long total_updates = params.width * params.height * params.steps;
    auto t0 = high_resolution_clock::now();
    while (params.steps > 0) {
        solver.step();
        params.steps--;
        solver.get_lattice();
    }
    auto t1 = high_resolution_clock::now();
    /* Round milliseconds as ints */
    auto elapsed = duration_cast<milliseconds>(t1 - t0); // elapsed time
    return total_updates * 1000 / elapsed.count();
}

int main() {
    Parameters params = load_parameters("../src/res/config.ini");
    Solver *solver;

    if (params.gpu) solver = new GpuSolver(params);
    else solver = new CpuSolver(params);

    switch (params.type) {
        case BENCHMARK: {
            std::cout << "Running a benchmark" << std::endl;
            float lattice_updates_per_seconds = run_benchmark(*solver, params);
            printf("Got %.f LUPS", lattice_updates_per_seconds);
        } break;

        case PARAVIEW:
            break;

        case REALTIME: {
            std::cout << "Running a realtime simulation" << std::endl;
            SdlEngine engine(*solver);
            engine.run();
        } break;

        default:
            std::cerr << "Simulation type not supported" << std::endl;
            break;
    }
}