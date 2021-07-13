#include <Lattice.hpp>
#include <cstdio>
#include <cmath>

double modulus2D(double x, double y);

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
                double e_dp_u = 0.0;
                for(int j = 0; j < D; j++) {
                    e_dp_u += flow_velocity(x, y, j) * e[i][j];
                }
                double mod_u = pow(flow_velocity(x, y, 0), 2) + pow(flow_velocity(x, y, 1), 2);
                density(x, y, i) = W[i]*(1+(3*e_dp_u)+(4.5*pow(e_dp_u, 2))-(1.5*mod_u));
            }
        }
    }
}

Lattice::~Lattice() { }

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

    for(int y = 0; y < HEIGHT; y++) {
        dest = (Uint32*)((Uint8*) pixels + y * pitch);
        for(int x = 0; x < WIDTH; x++) {
            unsigned int val = (unsigned int) std::min(modulus2D(flow_velocity(x, y, 0), flow_velocity(x, y, 1)) * 255, 255.0);
            *(dest + x) = ((0xFF000000|(val<<16)|(val<<8)|val));
        }
    }

    SDL_UnlockTexture(screen);
    step();
}

void Lattice::stream()
{
    for(int x = 1; x < WIDTH - 1; x++) {
        for(int y = 1; y < HEIGHT - 1; y++) {
            for(int i = 0; i < Q; i++) {
                density_t(x + e[i][0],y + e[i][1], i) = density(x, y, i);
            }
        }
    }
}

void Lattice::collide()
{

    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            /* Compute the total density of lattice site at position (x, y) */
            double total_density = 0.0;
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
                double e_dp_u = 0.0;
                for(int j = 0; j < D; j++) {
                    e_dp_u += flow_velocity(x, y, j) * e[i][j];
                }
                /* Compute modulus */
                double mod_u = pow(flow_velocity(x, y, 0), 2) + pow(flow_velocity(x, y, 1), 2);
                double density_eq = total_density * W[i] * (1 + (3*e_dp_u) + (4.5*pow(e_dp_u, 2)) - (1.5*mod_u));
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
double modulus2D(double x, double y)
{
    return sqrt((x*x) + (y*y));
}

/* HSL to RGB */
unsigned int hsl(unsigned int h, unsigned int s, unsigned int l)
{
    
}
