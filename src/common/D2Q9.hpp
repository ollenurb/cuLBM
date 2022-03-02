//
// Created by matteo on 7/20/21.
//
#pragma once

#include "Vector2D.hpp"
#include <cstdio>
#define Q 9
#define WEIGHTS {4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36}
#define E {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0,-1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}}

namespace D2Q9 {
/* Define a Lattice where:
 * f[i][j][q] = The macroscopic density at the position (i, j) with direction q
 * u[i][j] = velocity at position (i, j)
 * obstacle[i][j] = true if there is an obstacle at position (i, j), false otherwise
 */
  struct Lattice {
    unsigned WIDTH, HEIGHT;
    Real ***f;
    Vector2D<Real> **u;
    bool **obstacle;
    /* Allocate dynamic data structures */
    Lattice(unsigned w, unsigned h) : WIDTH(w), HEIGHT(h)  {
      u = new Vector2D<Real>*[WIDTH];
      obstacle = new bool*[WIDTH];
      f = new Real**[WIDTH];
      for(int i = 0; i < WIDTH; i++) {
        u[i] = new Vector2D<Real>[HEIGHT];
        obstacle[i] = new bool[HEIGHT];
        f[i] = new Real*[HEIGHT];
        for(int j = 0; j < HEIGHT; j++) {
          f[i][j] = new Real[Q];
        }
      }
    }

    /* Free resources */
    ~Lattice() {
      for(int i = 0; i < WIDTH; i++) {
        delete u[i];
        delete obstacle[i];
        for(int j = 0; j < HEIGHT; j++) {
          delete f[i][j];
        }
        delete f[i];
      }
      delete u;
      delete f;
      delete obstacle;
    }

    void swap(Lattice& lattice) {
      std::swap(this->f, lattice.f);
      std::swap(this->u, lattice.u);
    }
  };

  /* Define LBM constants */
  constexpr static const Vector2D<Real> VELOCITY = {0.070, 0.0};
  constexpr static const Real VISCOSITY = 0.020;
  constexpr static const Real OMEGA = 1 / (3 * VISCOSITY + 0.5);

  /* Allowed displacement vectors */
  /* "cast" into Vector2D so that it is more convenient to use later */
  constexpr static const Vector2D<int> e[Q] = E;

  /* Weights associated to each direction */
  const Real W[Q] = WEIGHTS;
}
