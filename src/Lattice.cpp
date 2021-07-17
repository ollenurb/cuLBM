#include "Lattice.hpp"
#include <cmath>
#include <algorithm>

unsigned int HSBtoRGB(float hue, float saturation, float brightness);

Lattice::Lattice(unsigned int w, unsigned int h):
    Renderizable(w, h),
    lattice(w, h),
    lattice_t(w, h)
{
    double e_dp_u;
    /* Initialize the initial configuration */
    initial_config.macroscopic_velocity.x = VELOCITY;
    initial_config.macroscopic_velocity.y = 0;
    for(int i = 0; i < Q; i++) {
        e_dp_u = e[i] * initial_config.macroscopic_velocity;
        initial_config.density[i] = initial_config.density_eq[i] = W[i] * (1 + (3*e_dp_u) + (4.5*(e_dp_u * e_dp_u)) - (1.5 * initial_config.macroscopic_velocity.mod_sqr()));
    }

    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            /* Initialize flow velocity */
            lattice(x, y) = initial_config;
            /* Initialize density function */
//            for(int i = 0; i < Q; i++) {
//                lattice(x, y)
//            }
        }
    }

    int size = 10;
    int x_center = WIDTH / 2 - (size / 2);
    int y_center = HEIGHT / 2 - (size / 2);

    for(int x = x_center; x < x_center + size; x++) {
        for (int y = y_center; y < y_center + size; y++) {
            for(int i = 0; i < Q; i++) {
                lattice(x, y).density[i] += .020;
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
            b = std::min(lattice(x, y).macroscopic_velocity.modulus() * 3.0, 1.0);
            *(dest + x) = HSBtoRGB(0.5, 1.0, b);
        }
    }
    SDL_UnlockTexture(screen);
}

inline unsigned clamp(unsigned val, unsigned low, unsigned high)
{
    return std::min(std::max(val, low), high);
}

void Lattice::stream()
{
    /* Check Horizontal Boundary conditions */
    LatticeNode &left = lattice_t(0, 0);
    LatticeNode &right = lattice_t(0, 0);

    for(int y = 0; y < HEIGHT; y++) {
        left = lattice_t(0, y);
        right = lattice_t(WIDTH-1, y);

        /* Left Wall */
        left.density[Ei] = left.density[Wi];
        left.density[SEi] = left.density[NWi];
        left.density[NEi] = left.density[SWi];

        left.density[Wi] = W[Wi] * (1 - 3*(VELOCITY) + 3*(VELOCITY * VELOCITY));
        left.density[NWi] = W[NWi] * (1 - 3*VELOCITY + 3*(VELOCITY * VELOCITY));
        left.density[SWi] = W[SWi] * (1 - 3*VELOCITY + 3*(VELOCITY * VELOCITY));

        /* Right Wall */
        right.density[Wi] =  right.density[Ei];
        right.density[NWi] = right.density[SEi];
        right.density[SWi] = right.density[NEi];

        right.density[Ei] = W[Ei] * (1 + 3*VELOCITY + 3*(VELOCITY * VELOCITY));
        right.density[SEi] = W[SEi] * (1 + 3*VELOCITY + 3*(VELOCITY * VELOCITY));
        right.density[NEi] = W[NEi] * (1 + 3*VELOCITY + 3*(VELOCITY * VELOCITY));
    }

    /* Check Vertical Boundary conditions */
    LatticeNode &top = lattice_t(0, 0);
    LatticeNode &bottom = lattice_t(0, 0);

    for(int x = 0; x < WIDTH; x++) {
        top = lattice_t(x, 0);
        bottom = lattice_t(x, HEIGHT-1);

        /* Top Wall */
        top.density[Si] = top.density[Ni];
        top.density[SWi] = top.density[NEi];
        top.density[SEi] = top.density[NWi];

        top.density[Ni] = W[Ni] * (1 - 1.5*(VELOCITY * VELOCITY));
        top.density[NEi] = W[NEi] * (1 + 3*VELOCITY + 3*(VELOCITY * VELOCITY));
        top.density[NWi] = W[NWi] * (1 - 3*VELOCITY + 3*(VELOCITY * VELOCITY));

        /* Bottom Wall */
        bottom.density[Ni] = bottom.density[Si];
        bottom.density[NWi] = bottom.density[SEi];
        bottom.density[NEi] = bottom.density[SWi];

        bottom.density[Si] = W[Si] * (1 - 1.5*(VELOCITY * VELOCITY));
        bottom.density[SEi] = W[SEi] * (1 + 3*VELOCITY + 3*(VELOCITY * VELOCITY));
        bottom.density[SWi] = W[SWi] * (1 - 3*VELOCITY + 3*(VELOCITY * VELOCITY));
    }
    /* Move the fluid to neighbouring sites */
    /* This doesn't work */
    unsigned x_index, y_index;

    for(int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) {
            for(int i = 0; i < Q; i++) {
                x_index = clamp(x + e[i].x, 0, WIDTH-1);
                y_index = clamp(y + e[i].y, 0, HEIGHT-1);
                lattice_t(x_index, y_index).density[i] = lattice(x, y).density[i];
            }
        }
    }

}

void Lattice::collide()
{
    double total_density;
    double e_dp_u;
    double mod_u;

    Vector2D u;

    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            LatticeNode &cur_node = lattice(x, y);
            /* Compute the total density of lattice site at position (x, y) */
            total_density = 0.0;
            u.x = u.y = 0.0;

            for(int i = 0; i < Q; i++) {
                total_density += cur_node.density[i];
                /* Accumulate the density inside each component of flow_velocity */
                u.x += cur_node.density[i] * e[i].x; // U_{x} component
                u.y += cur_node.density[i] * e[i].y; // U_{y} component
            }

            /* Compute average to get the actual value of flow_velocity */
            /* "Cast" to 0 if the velocity is negative */
            u.x = std::max(0.0, u.x / total_density);
            u.y = std::max(0.0, u.y / total_density);

            /* Compute densities at thermal equilibrium */
            /* Equation (8) */
            for(int i = 0; i < Q; i++) {
                e_dp_u = e[i] * u;
                cur_node.density_eq[i] = total_density * W[i] * (1 + (3*e_dp_u) + (4.5*(e_dp_u * e_dp_u)) - (1.5 * u.mod_sqr()));
                cur_node.density[i] += OMEGA * (cur_node.density_eq[i] - cur_node.density[i]);
            }
            cur_node.total_density = total_density;
            cur_node.macroscopic_velocity = u;
        }
    }
}

void Lattice::step()
{
    collide();
    stream();
    lattice.swap(lattice_t);
}

/* +============+ Vector2D functions +============+ */
/* They are const because they do not modify member's state */
inline double Vector2D::mod_sqr() const {
    return (x * x) + (y * y);
}

inline double Vector2D::modulus() const {
    return sqrt(mod_sqr());
}

double Vector2D::operator*(Vector2D &v) const {
    return (x * v.x) + (y * v.y);
}

/* +=========+ Functions no related to the class implementation +=========+ */
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

