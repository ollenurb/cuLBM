#ifdef SDL2
#include "common/engines/SdlEngine.hpp"
#endif
#include "cpu/CpuSolver.hpp"
#include "gpu/GpuSolver.cuh"
#include <chrono>
#include <iostream>

/*
 *  Run the solver for a given number of steps
 *  Returns the number of Lattice Updates Per Seconds
 */
void run_benchmark(Solver &solver, Parameters params) {
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;
    using std::chrono::seconds;

    unsigned long long total_updates = params.width * params.height * params.steps;
    unsigned steps = params.steps;
    auto t0 = high_resolution_clock::now();
    while (steps > 0) {
        solver.step();
        steps--;
    }
    solver.get_lattice();
    auto t1 = high_resolution_clock::now();
    /* Round milliseconds as ints */
    auto elapsed = duration_cast<milliseconds>(t1 - t0); // elapsed time
    float lups = total_updates * 1000 / elapsed.count();
    printf("LUPS: %f, Elapsed: %d,  Steps: %d\n", lups, elapsed.count(), params.steps);
}

int main() {
    Parameters params = load_parameters("../src/res/config.ini");
    Solver *solver;

    if (params.gpu) solver = new GpuSolver(params);
    else solver = new CpuSolver(params);

    switch (params.type) {
        case BENCHMARK: {
            std::cout << "Running a benchmark" << std::endl;
            run_benchmark(*solver, params);
        } break;

        case PARAVIEW:
            break;

        case REALTIME: {
#ifdef SDL2
            std::cout << "Running a realtime simulation" << std::endl;
            SdlEngine engine(*solver);
            engine.run();
#else
            std::cerr << "Realtime simulation is not supported since SDL2 wasn't found during compilation" << std::endl;
#endif
        } break;

        default:
            std::cerr << "Simulation type not supported" << std::endl;
            break;
    }
}