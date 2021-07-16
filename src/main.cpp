#include "Engine.hpp"
#include "Lattice.hpp"

#define D 2
#define Q 9
#define WIDTH 600
#define HEIGHT 240

int main(int argc, char** argv)
{
    /* === SDL-Related variables === */
    Lattice lattice(WIDTH, HEIGHT);
//    for(int i = 0; i < 3; i++) {
//        std::cout << "Step " << i;
//        lattice.step();
//    }
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
