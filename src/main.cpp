#include <SDL2/SDL.h>
#include <Engine.hpp>
#include <Lattice.hpp>

#define D 2
#define Q 9
#define WIDTH 600
#define HEIGHT 500

int main(int argc, char** argv)
{
    /* === SDL-Related variables === */
    Lattice lattice(WIDTH, HEIGHT);
    /* printf("Step 1's velocity:\n"); */
    /* lattice.step(); */
    /* printf("Step 2's velocity:\n"); */
    /* lattice.step(); */
    /* printf("Step 3's velocity:\n"); */
    /* lattice.step(); */
    /* printf("Step 4's velocity:\n"); */
    /* lattice.step(); */
    Engine engine(lattice);
    engine.run();

}
