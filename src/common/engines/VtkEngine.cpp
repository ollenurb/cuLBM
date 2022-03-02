//
// Created by matteo on 9/16/21.
//

#include "VtkEngine.hpp"
#include <cstdio>

VtkEngine::VtkEngine(Simulation &s, unsigned int n_steps) : WIDTH(s.get_width()), HEIGHT(s.get_height()), STEPS(n_steps), simulation(s) {}

VtkEngine::~VtkEngine() = default;

void VtkEngine::run() {
  unsigned int cur_step = 0;
  unsigned int update_counter = 0;
  char f_name[80];
  FILE *fp;

  while (cur_step <= STEPS) {
    // IDEA: Update only after 50 steps
    simulation.step();
    if (update_counter == 50) {
      sprintf(f_name, "img/%d_step.vtk", cur_step);
      fp = fopen(f_name, "w");
      if(fp == nullptr) {
        // TODO: throw error
      }

      // Write header, then step with simulation then render
      write_header(fp);
      write_data(fp, simulation.get_lattice());
      fclose(fp);
      printf("[%2u%%] (%d/%d)\n", (cur_step*100)/STEPS, cur_step, STEPS);
      cur_step += 1;
      update_counter = 0;
    }
    update_counter++;
  }
}

void VtkEngine::write_data(FILE *fp, const D2Q9::Lattice* lattice) const {
  // Write velocity
  fprintf(fp,"POINT_DATA %d\n", WIDTH*HEIGHT);
  fprintf(fp, "VECTORS velocity float\n");
  for(int x = 0; x < WIDTH; x++) {
    for(int y = 0; y < HEIGHT; y++) {
      const Vector2D<Real> current = lattice->u[x][y];
      fprintf(fp, "%f %f %f\n", current.x, current.y, 0.0);
    }
  }
  // TODO: Add curl
}

void VtkEngine::write_header(FILE *file) const {
  if(file == nullptr) {
    // TODO: Throw error
  }

  // Write preamble
  fprintf(file,"# vtk DataFile Version 2.0\n");
  fprintf(file,"generated by LBM simulator (written by Brunello Matteo) \n");
  fprintf(file,"ASCII\n");
  fprintf(file,"DATASET STRUCTURED_GRID\n");
  fprintf(file,"DIMENSIONS  %d %d 1\n", WIDTH, HEIGHT);
  fprintf(file,"POINTS %d float\n", WIDTH * HEIGHT);

  // Write point coordinates
  for(int x = 0; x < WIDTH; x++) {
    for(int y = 0; y < HEIGHT; y++) {
      fprintf(file, "%f %f %f\n",(1.0) * x, (1.0) * y, 0.0);
    }
  }
}

