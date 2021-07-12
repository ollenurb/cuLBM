#include <SDL2/SDL.h>
#include <Engine.hpp>
#include <Lattice.hpp>

#define D 2
#define Q 9
#define WIDTH 100
#define HEIGHT 100

#define D_T 1
#define D_X 1

const float viscosity = 0.02;
const float omega = (1/3)*((1/viscosity)+6);
/* float avg[WIDTH][HEIGHT][D]; */

/* Allowed displacement vectors */
const int e[Q][D] =
{
    { 0, 0}, { 1,  0}, {0,  1},
    {-1, 0}, { 0, -1}, {1,  1},
    {-1, 1}, {-1, -1}, {1, -1}
};

/* Probabilities associated to each lattice direction (aka. Weights) */
const float w[Q] = {4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36};

float dot_product(float *a, float *b, int dim)
{
    float result = 0.0;
    for(int i = 0; i < dim; i++) {
        result += a[i]*b[i];
    }
    return result;
}

void collide(float ***density)
{
    float density_eq[Q][WIDTH][HEIGHT];
    /* Used to store results */
    float avg[WIDTH][HEIGHT][D];
    float tot = 0.0f;
    for(int x = 0; x < WIDTH; x++) {
        for(int y = 0; y < HEIGHT; y++) {
            for(int i = 0; i < Q; i++) {
                /* For each lattice site */
                /* Compute total density */
                tot += density[i][x][y];
                /* Accumulate sum for average */
                for(int d = 0; d < D; d++) {
                    avg[x][y][d] += density[i][x][y] * e[i][d]; // U_x,y components
                }
            }
            /* Compute average */
            for(int d = 0; d < D; d++) {
                avg[x][y][d] = avg[x][y][d] / tot;
            }

            /* Compute densities at thermal equilibrium */
            for(int i = 0; i < Q; i++) { /* For each lattice site */
                float e_dp_u = dot_product((float *) e[i], avg[x][y], D);
                float mod_u = pow(avg[x][y][0],2) + pow(avg[x][y][1],2);
                density_eq[i][x][y] = tot*w[i]*(1+(3*e_dp_u)+((9/2)*pow(e_dp_u,2))-((3/2)*mod_u));

                density[i][x][y] = omega * (density[i][x][y] - density_eq[i][x][y]);
            }
        }
    }
}

void stream(float ***density, float ***density_t)
{
    for(int x = 1; x < WIDTH - 1; x++) {
        for(int y = 1; y < HEIGHT - 1; y++) {
            for(int i = 0; i < Q; i++) {
                density_t[i][x + e[i][0]][y + e[i][1]] = density[i][x][y];
            }
        }
    }
}

void step(float ***density, float ***density_t)
{
    float ***tmp = density;
    collide(density);
    stream(density, density_t);
    density = density_t;
    density_t = tmp;
}

float ***create_2d_lattice(int q, int width, int height) {

    float ***ret_lattice = (float ***) malloc(sizeof(float **) * q);
    for(int i = 0; i < q; i++) {
        ret_lattice[i] = (float **) malloc(sizeof(float *) * width);
        for(int x = 0; x < width; x++) {
            ret_lattice[i][x] = (float *) malloc(sizeof(float) * height);
        }
    }
    return ret_lattice;
}

void speed_to_rgb(float speed, SDL_Color& color)
{
}


float modulus2D(float v[2])
{
    return sqrt((v[0] * v[0]) + (v[1] * v[1]));
}

// TODO: Rename function
void simulation_step(SDL_Texture *screen)
{
    /* From SO: void **pixels is a pointer-to-a-pointer; these are typically
     * used (in this kind of context) where the data is of a pointer type but
     * memory management is handled by the function you call. */
    void *pixels;
    int pitch;
    Uint32 *dest;

    float speed;

    if (SDL_LockTexture(screen, NULL, &pixels, &pitch) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't lock texture: %s\n", SDL_GetError());
    }

    for(int y = 0; y < HEIGHT; y++) {
        dest = (Uint32*)((Uint8*) pixels + y * pitch);
        for(int x = 0; x < WIDTH; x++) {
            /* speed = modulus2D(avg[x][y]); */
            *(dest + x) = ((0xFF000000|(121<<16)|(255<<8)|20));
        }
    }

    SDL_UnlockTexture(screen);
}


int main(int argc, char** argv)
{
    /* A lattice consists of a 2D grid with 9 velocity vectors */
    /* float ***density = create_2d_lattice(Q, WIDTH, HEIGHT); */
    /* float ***density_t = create_2d_lattice(Q, WIDTH, HEIGHT); */

    /* Initialize the lattice */
    /* for(int x = 0; x < WIDTH; x++) { */
    /*     for(int y = 0; y < HEIGHT; y++) { */
    /*         for(int i = 0; i < Q; i++) { */
    /*             density[i][x][y] = 1.0; */
    /*             density_t[i][x][y] = 0.0; */
    /*         } */
    /*     } */
    /* } */

    /* === SDL-Related variables === */

    Lattice lattice(WIDTH, HEIGHT);
    Engine engine(lattice);
}
