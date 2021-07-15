#include "Lattice.hpp"
#include <cstdio>
#include <cmath>

inline double modulus2D(double x, double y);
inline double sqr(double a);
unsigned int HSBtoRGB(float hue, float saturation, float brightness);

Lattice::Lattice(unsigned int w, unsigned int h):
    Renderizable(w, h),
    flow_velocity(w, h, D),
    density(w, h, Q),
    density_t(w, h, Q)
{
    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            /* Initialize flow velocity */
            flow_velocity(x, y, 0) = VELOCITY;
            flow_velocity(x, y, 1) = 0;
            /* Initialize density function */
            for(int i = 0; i < Q; i++) {
                density(x, y, i) = W[i];
            }
        }
    }
}

Lattice::~Lattice() = default;

void Lattice::render(SDL_Texture* screen)
{
    /* From Stack Overflow: void **pixels is a pointer-to-a-pointer; these are
     * typically used (in this kind of context) where the data is of a pointer
     * type but memory management is handled by the function you call.
     */
    void *pixels;
    int pitch;
    Uint32 *dest;
    float b;

    if (SDL_LockTexture(screen, nullptr, &pixels, &pitch) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't lock texture: %s\n", SDL_GetError());
    }

    for(int y = 0; y < HEIGHT; y++) {
        dest = (Uint32*)((Uint8*) pixels + y * pitch);
        for(int x = 0; x < WIDTH; x++) {
            b = std::min(modulus2D(flow_velocity(x, y, 0), flow_velocity(x, y, 1)) * 3.0, 1.0);
            *(dest + x) = HSBtoRGB(0.5, 1.0, b);
        }
    }
    SDL_UnlockTexture(screen);
}

void Lattice::stream()
{
    /* Move the fluid to neighbouring sites */
    for(int x = 1; x < WIDTH - 1; x++) {
        for(int y = 1; y < HEIGHT - 1; y++) {
            for(int i = 0; i < Q; i++) {
                density_t(x + e[i][0], y + e[i][1], i) = density(x, y, i);
            }
        }
    }

    /* Check Horizontal Boundary conditions */
    /* (On-Grid bounce back) */
    /* TODO: Has to be fixed */
    for(int y = 0; y < HEIGHT; y++) {
        density_t(0, y, 1) += density_t(0, y, 3);
        density_t(0, y, 8) += density_t(0, y, 6);
        density_t(0, y, 5) += density_t(0, y, 7);

        density_t(0, y, 3) = 0;
        density_t(0, y, 6) = 0;
        density_t(0, y, 7) = 0;

        density_t(WIDTH-1, y, 3) += density_t(WIDTH-1, y, 1);
        density_t(WIDTH-1, y, 6) += density_t(WIDTH-1, y, 8);
        density_t(WIDTH-1, y, 7) += density_t(WIDTH-1, y, 5);

        density_t(WIDTH-1, y, 1) = 0;
        density_t(WIDTH-1, y, 5) = 0;
        density_t(WIDTH-1, y, 8) = 0;
    }

    for(int x = 0; x < WIDTH; x++) {
        density_t(x, 0, 4) += density_t(x, 0, 2);
        density_t(x, 0, 7) += density_t(x, 0, 6);
        density_t(x, 0, 8) += density_t(x, 0, 5);

        density_t(x, 0, 2) = 0;
        density_t(x, 0, 6) = 0;
        density_t(x, 0, 5) = 0;

        density_t(x, HEIGHT-1, 2) += density_t(x, HEIGHT-1, 4);
        density_t(x, HEIGHT-1, 6) += density_t(x, HEIGHT-1, 7);
        density_t(x, HEIGHT-1, 5) += density_t(x, HEIGHT-1, 8);

        density_t(x, HEIGHT-1, 4) = 0;
        density_t(x, HEIGHT-1, 7) = 0;
        density_t(x, HEIGHT-1, 8) = 0;

    }
}

void Lattice::collide()
{
    double total_density;
    double e_dp_u;
    double density_eq;
    double mod_u;

    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            /* Compute the total density of lattice site at position (x, y) */
            total_density = 0.0;
            flow_velocity(x, y, 0) = flow_velocity(x, y, 1) = 0.0;

            for(int i = 0; i < Q; i++) {
                total_density += density(x, y, i);
                /* Accumulate the density inside each component of flow_velocity */
                flow_velocity(x, y, 0) += density(x, y, i) * e[i][0]; // U_{x} component
                flow_velocity(x, y, 1) += density(x, y, i) * e[i][1]; // U_{y} component
            }
            /* Compute average to get the actual value of flow_velocity */
            /* "Cast" to 0 if the velocity is negative */
            flow_velocity(x, y, 0) = std::max(0.0, flow_velocity(x, y, 0) / total_density);
            flow_velocity(x, y, 1) = std::max(0.0, flow_velocity(x, y, 1) / total_density);

            /* Compute densities at thermal equilibrium */
            /* Equation (8) */
            for(int i = 0; i < Q; i++) {
                /* Compute dot product */
                e_dp_u = 0.0;
                for(int j = 0; j < D; j++) {
                    e_dp_u += flow_velocity(x, y, j) * e[i][j];
                }
                /* Compute modulus */
                mod_u = sqr(flow_velocity(x, y, 0)) + sqr(flow_velocity(x, y, 1));
                density_eq = total_density * W[i] * (1 + (3*e_dp_u) + (4.5*sqr(e_dp_u)) - (1.5*mod_u));
                /* Equation (9) */
                density(x, y, i) += OMEGA * (density_eq - density(x, y, i));
            }
        }
    }
}

void Lattice::step()
{
    collide();
    stream();
    density.swap(density_t);
}


/* +=========+ Functions no related to the class implementation +=========+ */
/* TODO: Please, find a more suitable organization here */
inline double modulus2D(double x, double y)
{
    return sqrt((x*x) + (y*y));
}


inline double sqr(double a)
{
    return a * a;
}

unsigned int HSBtoRGB(float hue, float saturation, float brightness) {
    int r = 0, g = 0, b = 0;
    if (saturation == 0) {
        r = g = b = (int) (brightness * 255.0f + 0.5f);
    } else {
        float h = (hue - (float) floor(hue)) * 6.0f;
        float f = h - (float) floor(h);
        float p = brightness * (1.0f - saturation);
        float q = brightness * (1.0f - saturation * f);
        float t = brightness * (1.0f - (saturation * (1.0f - f)));
        switch ((unsigned) h) {
            case 0:
                r = (unsigned) (brightness * 255.0f + 0.5f);
                g = (unsigned) (t * 255.0f + 0.5f);
                b = (unsigned) (p * 255.0f + 0.5f);
                break;
            case 1:
                r = (unsigned) (q * 255.0f + 0.5f);
                g = (unsigned) (brightness * 255.0f + 0.5f);
                b = (unsigned) (p * 255.0f + 0.5f);
                break;
            case 2:
                r = (unsigned) (p * 255.0f + 0.5f);
                g = (unsigned) (brightness * 255.0f + 0.5f);
                b = (unsigned) (t * 255.0f + 0.5f);
                break;
            case 3:
                r = (unsigned) (p * 255.0f + 0.5f);
                g = (unsigned) (q * 255.0f + 0.5f);
                b = (unsigned) (brightness * 255.0f + 0.5f);
                break;
            case 4:
                r = (unsigned) (t * 255.0f + 0.5f);
                g = (unsigned) (p * 255.0f + 0.5f);
                b = (unsigned) (brightness * 255.0f + 0.5f);
                break;
            case 5:
                r = (unsigned) (brightness * 255.0f + 0.5f);
                g = (unsigned) (p * 255.0f + 0.5f);
                b = (unsigned) (q * 255.0f + 0.5f);
                break;
        }
    }
    return ((0xFF000000|(r<<16)|(g<<8)|b));
}
