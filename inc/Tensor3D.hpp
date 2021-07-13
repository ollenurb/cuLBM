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
};
