#include <iostream>
#include "common/engines/SdlEngine.hpp"
#include "cpu/CpuSolver.hpp"

int main() {
    Parameters params = load_parameters("../src/res/config.ini");
    printf("Loaded %d, %d", params.width, params.height);
    Solver *solver = new CpuSolver(params);
    SdlEngine engine(*solver);
    engine.run();
}