#include <Lattice.hpp>
#include <cstdio>
#include <cmath>

Lattice::Lattice(unsigned int w, unsigned int h)
    : Renderizable(w, h), flow_velocity(w, h, D)
{
    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            flow_velocity(x, y, 0) = x + y;
            flow_velocity(x, y, 1) = x + y;
        }
    }
}

Lattice::~Lattice()
{
}

double modulus2D(double x, double y)
{
    return sqrt((x*x) + (y*y));
}

void Lattice::render(SDL_Texture* screen)
{
    /* From Stack Overflow: void **pixels is a pointer-to-a-pointer; these are
     * typically used (in this kind of context) where the data is of a pointer
     * type but memory management is handled by the function you call.
     */
    void *pixels;
    int pitch;
    Uint32 *dest;

    if (SDL_LockTexture(screen, NULL, &pixels, &pitch) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't lock texture: %s\n", SDL_GetError());
    }

    /* TODO Change this to display useful informations */
    for(int y = 0; y < HEIGHT; y++) {
        dest = (Uint32*)((Uint8*) pixels + y * pitch);
        for(int x = 0; x < WIDTH; x++) {


            unsigned int val = ((unsigned int)modulus2D(flow_velocity(x, y, 0), flow_velocity(x, y, 1))) % 255;
            *(dest + x) = ((0xFF000000|(val<<16)|(val<<8)|val));
        }
    }

    SDL_UnlockTexture(screen);
}

