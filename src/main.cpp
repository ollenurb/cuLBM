#include "common/Engine.hpp"
#include "cpu/LBM.hpp"
#include <chrono>
#include <iostream>

#define WIDTH 600
#define HEIGHT 240

void run_benchmark(unsigned long steps)
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    std::cout << "Simulating " << steps << " time steps of a "
              << WIDTH << "x" << HEIGHT << " lattice"
              << std::endl;

    LBM lattice(WIDTH, HEIGHT);
    auto t0 = high_resolution_clock::now();
    while(steps > 0) {
        lattice.step();
        steps--;
    }
    auto t1 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer */
    auto ms_int = duration_cast<milliseconds>(t1 - t0);

    std::cout << "The program took " << ms_int.count() << "ms to complete" << std::endl;
}

int main(int argc, char** argv)
{
    LBM lattice(WIDTH, HEIGHT);
    Engine engine(lattice);
    engine.run();
//    run_benchmark(100);
}
