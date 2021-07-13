#include <SDL2/SDL.h>
#include <Engine.hpp>
#include <Lattice.hpp>

#define D 2
#define Q 9
#define WIDTH 1000
#define HEIGHT 600

int main(int argc, char** argv)
{
    /* === SDL-Related variables === */
    Lattice lattice(WIDTH, HEIGHT);
    /* printf("\nStep 1's velocity:\n"); */
    /* lattice.step(); */
    /* printf("\nStep 2's velocity:\n"); */
    /* lattice.step(); */
    /* printf("\nStep 2's velocity:\n"); */
    /* lattice.step(); */
    /* printf("\nStep 2's velocity:\n"); */
    /* lattice.step(); */

    /* /1* printf("Step 2's velocity:\n"); *1/ */
    /* /1* lattice.step(); *1/ */
    /* /1* printf("Step 3's velocity:\n"); *1/ */
    /* /1* lattice.step(); *1/ */
    /* /1* printf("Step 4's velocity:\n"); *1/ */
    /* /1* lattice.step(); *1/ */
    Engine engine(lattice);
    engine.run();

}
