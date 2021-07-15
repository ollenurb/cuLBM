#pragma once
#include <stdexcept>
#include <cstdio>

template <typename T>
class Tensor3D
{
    private:
    const unsigned int X;
    const unsigned int Y;
    const unsigned int Z;
    const unsigned int SZ;
    T *data;

    public:
    Tensor3D(unsigned int x, unsigned int y, unsigned int z) : X(x), Y(y), Z(z), SZ(x * y * z) { data = new T[SZ]; }
    ~Tensor3D() { delete data; }

    T& operator()(unsigned int i, unsigned int j, unsigned int k)
    {
        unsigned int index = i * Y * Z + j * Z + k;
        if(index > SZ) {
            throw std::invalid_argument("Tensor3D: Index out of bounds");
        }
        return data[index];
    }

    // TODO: Insert control on sizes
    void swap(Tensor3D& swappable)
    {
        T *tmp = swappable.data;
        swappable.data = data;
        data = tmp;
    }

    void print()
    {
        /* for(int y = 0; y < Y; y++) { */
        /*     for(int x = 0; x < X; x++) { */
        /*         printf("{ "); */
        /*         for(int z = 0; z < Z; z++) { */
        /*             printf("%.8f ", data[y * Y * Z + x * Z + z]); */
        /*         } */
        /*         printf("} "); */
        /*     } */
        /*     printf("\n"); */
        /* } */
        for (int i = 0; i < X*Y*Z; i++) {
            printf("%.8f ", data[i]);
        }
    }

};
