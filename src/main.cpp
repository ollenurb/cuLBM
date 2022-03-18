#include <iostream>
#include "common/engines/SdlEngine.hpp"
#include "cpu/CpuSolver.hpp"
#include "gpu/GpuSolver.cuh"

int main() {
    Parameters params = load_parameters("../src/res/config.ini");
//    printf("Loaded %d, %d", params.width, params.height);
    Solver *solver = new GpuSolver(params);
    SdlEngine engine(*solver);
    engine.run();
}