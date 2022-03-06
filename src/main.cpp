#include <chrono>
#include <iostream>
#include "../lib/CLI11.hpp"
#include "common/engines/SdlEngine.hpp"
#include "common/engines/VtkEngine.hpp"
#include "cpu/CpuSolver.hpp"

void run_benchmarks(Solver &simulation, unsigned int steps) {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  auto t0 = high_resolution_clock::now();
  while (steps > 0) {
    simulation.step();
    steps--;
  }
  auto t1 = high_resolution_clock::now();

  /* Getting number of milliseconds as an integer */
  auto ms_int = duration_cast<milliseconds>(t1 - t0);

  std::cout << "The program took " << ms_int.count() << "ms to complete" << std::endl;
}

int main(int argc, char **argv) {
  CLI::App app("Lattice Boltzmann Method CFD Solver");
  enum ProgramMode {
    BENCHMARK, REALTIME, PARAVIEW
  };
  static const char *mode_str[] = {"benchmark", "realtime simulation", "ParaView simulation"};

  bool gpu_support = false;
  enum ProgramMode mode = REALTIME;
  std::pair<unsigned, unsigned> dim(100, 100);
  unsigned int steps = 10000;
  app.add_option("--gpu", gpu_support, "Whether to use GPU acceleration or not (Default false)");
  app.add_option("--mode", mode,
                 "Run the program on a given mode. Available values are:\n\t0: Benchmark\n\t1: Realtime simulation (Default)\n\t2: ParaView simulation");
  app.add_option("--dim", dim, "Dimensions of the simulation expressed as WIDTH x HEIGHT (default 100 100)");
  app.add_option("--step", steps, "Number of time steps to be performed in the simulation (default 10000)");

  CLI11_PARSE(app, argc, argv)
  std::cout << "Running a "
            << mode_str[mode] << " on a "
            << dim.first << "x" << dim.second << " grid "
            << (gpu_support ? "with" : "without")
            << " GPU acceleration enabled" << std::endl;

  Solver *simulation;
  simulation = new CpuSolver(dim.first, dim.second);

  switch (mode) {
    case BENCHMARK: {
      run_benchmarks(*simulation, steps);
      break;
    }

    case REALTIME: {
      SdlEngine engine(*simulation);
      engine.run();
      break;
    }

    case PARAVIEW: {
      VtkEngine engine(*simulation, steps);
      engine.run();
      break;
    }
  }

}
