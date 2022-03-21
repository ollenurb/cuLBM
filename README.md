# cuLBM: A GPU accellerated Lattice Boltzmann Fluid solver

<p align="center">
  <img src="https://github.com/ollenurb/cuLBM/blob/main/img/Peek%202022-03-18%2009-01.gif" />
</p>

cuLBM was developed as a final project of a graduate (master) [course](http://magistrale.educ.di.unito.it/index.php/offerta-formativa/insegnamenti/elenco-completo/elenco-completo/scheda-insegnamento?cod=MFN0795&codA=&year=2020&orienta=YXFH). For this reason, it is not appropriate for serious scientific modeling or simulation.

cuLBM provides two implementations (one with GPU support and one without) of a 2D fluid solver based on the [Lattice-Boltzmann method](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods). The goal of the project is to apply parallelization techniques discussed in the course, and to empirically evaluate speedups.

## Installation
In order to compile the program, you need to have installed in your system `CUDA Toolkit` and `cmake`. If the requirements are satisfied, just clone this repository, then `cmake install` inside. 

## Program modes and configuration
The program can run in 3 different modes, which can be set on the `config.ini` file under `res`:
* **`paraview`**: writes the fluid solution in a .vtk file at the end of the program.
* **`realtime`**: uses SDL2 to simulate the fluid in real time.
* **`benchmark`**: calculates the number of grid updates per second of the solver.

The configuration file `config.ini` is also used to set up other parameters of the simulation:
* `width`: width of the simulation space
* `height`: height of the simulation space
* `velocity`: fluid velocity on each lattice site at `t=0`
* `viscosity`: fluid viscosity
* `gpu`: set if the simulation must be run with hardware acceleration

## Report
The final report (written in Italian) can be found inside the `report` directory. To generate the `pdf`, install `pandoc` and run `make`. 
