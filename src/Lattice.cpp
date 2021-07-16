#include "Lattice.hpp"
#include <cstdio>
#include <cmath>

inline double modulus2D(double x, double y);
inline double sqr(double a);
unsigned int HSBtoRGB(float hue, float saturation, float brightness);

Lattice::Lattice(unsigned int w, unsigned int h):
    Renderizable(w, h),
    lattice(w, h),
    lattice_t(w, h)
{
    double e_dp_u;
    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            /* Initialize flow velocity */
            lattice(x, y).macroscopic_velocity.x = VELOCITY;
            lattice(x, y).macroscopic_velocity.y = 0;
            /* Initialize density function */
            for(int i = 0; i < Q; i++) {
                e_dp_u = e[i] * lattice(x, y).macroscopic_velocity;
                lattice(x, y).density_eq[i] = W[i] * (1 + (3*e_dp_u) + (4.5*(e_dp_u * e_dp_u)) - (1.5 * lattice(x, y).macroscopic_velocity.mod_sqr()));
                lattice(x, y).density[i] = lattice_t(x, y).density[i] = lattice_t(x, y).density_eq[i] = lattice(x, y).density_eq[i];
            }
        }
    }

    for(int x = 100; x < 200; x++) {
        for (int y = 100; y < 200; y++) {
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

void Lattice::stream()
{
    /* Move the fluid to neighbouring sites */
    /* This doesn't work */
    for(int x = 1; x < WIDTH - 1; x++) {
        for(int y = 1; y < HEIGHT - 1; y++) {
            for(int i = 0; i < Q; i++) {
                lattice_t(x + e[i].x, y + e[i].y).density[i] = lattice(x, y).density[i];
            }
        }
    }

    /* Check Horizontal Boundary conditions */
    /* (On-Grid bounce back) */
    /* TODO: Has to be fixed */
//    for(int y = 0; y < HEIGHT; y++) {
//        density_t(0, y, 1) += density_t(0, y, 3);
//        density_t(0, y, 8) += density_t(0, y, 6);
//        density_t(0, y, 5) += density_t(0, y, 7);
//
//        density_t(0, y, 3) = 0;
//        density_t(0, y, 6) = 0;
//        density_t(0, y, 7) = 0;
//
//        density_t(WIDTH-1, y, 3) += density_t(WIDTH-1, y, 1);
//        density_t(WIDTH-1, y, 6) += density_t(WIDTH-1, y, 8);
//        density_t(WIDTH-1, y, 7) += density_t(WIDTH-1, y, 5);
//
//        density_t(WIDTH-1, y, 1) = 0;
//        density_t(WIDTH-1, y, 5) = 0;
//        density_t(WIDTH-1, y, 8) = 0;
//    }
//
//    for(int x = 0; x < WIDTH; x++) {
//        density_t(x, 0, 4) += density_t(x, 0, 2);
//        density_t(x, 0, 7) += density_t(x, 0, 6);
//        density_t(x, 0, 8) += density_t(x, 0, 5);
//
//        density_t(x, 0, 2) = 0;
//        density_t(x, 0, 6) = 0;
//        density_t(x, 0, 5) = 0;
//
//        density_t(x, HEIGHT-1, 2) += density_t(x, HEIGHT-1, 4);
//        density_t(x, HEIGHT-1, 6) += density_t(x, HEIGHT-1, 7);
//        density_t(x, HEIGHT-1, 5) += density_t(x, HEIGHT-1, 8);
//
//        density_t(x, HEIGHT-1, 4) = 0;
//        density_t(x, HEIGHT-1, 7) = 0;
//        density_t(x, HEIGHT-1, 8) = 0;
//
//    }
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
//            e_dp_u = u.x + u.y;

            /* Compute average to get the actual value of flow_velocity */
            /* "Cast" to 0 if the velocity is negative */
            u.x = std::max(0.0, u.x / total_density);
            u.y = std::max(0.0, u.y / total_density);

            /* Compute densities at thermal equilibrium */
            /* Equation (8) */
            for(int i = 0; i < Q; i++) {
                e_dp_u = e[i] * u;
                /* Compute modulus */
                cur_node.density_eq[i] = total_density * W[i] * (1 + (3*e_dp_u) + (4.5*(e_dp_u * e_dp_u)) - (1.5 * u.mod_sqr()));
                /* Equation (9) */
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
    return x*v.x + y*v.y;
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

