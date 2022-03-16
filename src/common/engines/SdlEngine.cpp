#include "SdlEngine.hpp"
#include <iostream>

#define UPDATE_STEPS 20

/* Utility function used to render */
unsigned int HSBtoRGB(float hue, float saturation, float brightness) {
    int r = 0, g = 0, b = 0;
    if (saturation == 0) {
        r = g = b = std::floor(brightness * 255.0f + 0.5f);
    } else {
        float h = (hue - (float) std::floor(hue)) * 6.0f;
        float f = h - (float) std::floor(h);
        float p = brightness * (1.0f - saturation);
        float q = brightness * (1.0f - saturation * f);
        float t = brightness * (1.0f - (saturation * (1.0f - f)));
        switch ((unsigned) h) {
            case 0:
                r = std::floor(brightness * 255.0f + 0.5f);
                g = std::floor(t * 255.0f + 0.5f);
                b = std::floor(p * 255.0f + 0.5f);
                break;
            case 1:
                r = std::floor(q * 255.0f + 0.5f);
                g = std::floor(brightness * 255.0f + 0.5f);
                b = std::floor(p * 255.0f + 0.5f);
                break;
            case 2:
                r = std::floor(p * 255.0f + 0.5f);
                g = std::floor(brightness * 255.0f + 0.5f);
                b = std::floor(t * 255.0f + 0.5f);
                break;
            case 3:
                r = std::floor(p * 255.0f + 0.5f);
                g = std::floor(q * 255.0f + 0.5f);
                b = std::floor(brightness * 255.0f + 0.5f);
                break;
            case 4:
                r = std::floor(t * 255.0f + 0.5f);
                g = std::floor(p * 255.0f + 0.5f);
                b = std::floor(brightness * 255.0f + 0.5f);
                break;
            case 5:
                r = std::floor(brightness * 255.0f + 0.5f);
                g = std::floor(p * 255.0f + 0.5f);
                b = std::floor(q * 255.0f + 0.5f);
                break;
        }
    }
    return ((0xFF000000 | (r << 16) | (g << 8) | b));
}

SdlEngine::SdlEngine(Solver &r) : simulation(r) {
    params = r.params;
    running = false;
    SDL_CreateWindowAndRenderer(params.width, params.height, 0, &window, &renderer);
    screen = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, params.width, params.height);
}

SdlEngine::~SdlEngine() = default;

void SdlEngine::run() {
    unsigned n_frame = 0;
    running = true;
    while (running) {
        process_events();
        /* TODO: Change 10 with AFTER_N_FRAMES */
        if (n_frame == UPDATE_STEPS) {
            render(screen);
            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, screen, nullptr, nullptr);
            SDL_RenderPresent(renderer);
            n_frame = 0;
            SDL_Delay(60);
        }
        simulation.step();
        n_frame++;
    }
}

void SdlEngine::process_events() {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_KEYDOWN:
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                }
                break;
            case SDL_QUIT:
                running = false;
                break;
        }
    }
}

void SdlEngine::render(SDL_Texture *) {
    Lattice<Host> lattice = simulation.get_lattice();
    Bitmap<Host> obstacle = simulation.get_obstacle();
    void *pixels;
    int pitch;
    Uint32 *dest;
    Real b;

    if (SDL_LockTexture(screen, nullptr, &pixels, &pitch) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't lock texture: %s\n", SDL_GetError());
    }

    for (int y = 0; y < params.height; y++) {
        dest = (Uint32 *) ((Uint8 *) pixels + y * pitch);
        for (int x = 0; x < params.width; x++) {
            b = std::min(lattice.u(x, y).modulus() * 3, static_cast<Real>(1));
            if (obstacle(x, y)) {
                *(dest + x) = ((0xFF000000 | (112 << 16) | (0 << 8) | 0));
            } else {
                *(dest + x) = HSBtoRGB(0.5, 1, b);
            }
        }
    }
    SDL_UnlockTexture(screen);
}
