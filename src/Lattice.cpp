#include <Lattice.hpp>
#include <cstdio>

Lattice::Lattice(unsigned int w, unsigned int h) : Renderizable(h, w) {
    printf("Lattice: Got (%d, %d)", w, h);
}

Lattice::~Lattice() { }
void Lattice::render(SDL_Texture* screen)
{
    /* From SO: void **pixels is a pointer-to-a-pointer; these are typically
     * used (in this kind of context) where the data is of a pointer type but
     * memory management is handled by the function you call. */
    void *pixels;
    int pitch;
    Uint32 *dest;

    if (SDL_LockTexture(screen, NULL, &pixels, &pitch) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't lock texture: %s\n", SDL_GetError());
    }

    /* TODO Chang this to display useful informations */
    for(int y = 0; y < HEIGHT; y++) {
        dest = (Uint32*)((Uint8*) pixels + y * pitch);
        for(int x = 0; x < WIDTH; x++) {
            *(dest + x) = ((0xFF000000|(121<<16)|(255<<8)|20));
        }
    }

    SDL_UnlockTexture(screen);
}

