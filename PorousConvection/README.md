# PorousConvection

[![Build Status](https://github.com/kszenes/pde-on-gpu-szenes/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kszenes/pde-on-gpu-szenes/actions/workflows/CI.yml?query=branch%3Amain)

## Theory

### Introduction
In this project, we implement a 3D multi-XPU solver for the convection of a fluid due to temperature through a porous media. This is a procees that is of particular interest when modelling geophysics.

### Equations


### Numerical Methods

## Results

### Porous Convection 2D
(Quiver arrows too large however in order to save node-hours simulation was not rerun.)
![porous-convection-2d](docs/2d_porous_long.gif)

### Porous Convection 3D

#### Final Timestep
##### 3D plot
![porous-convection-3d](docs/T_3D.png)
##### 2D slice
![porous-convection-3d-slice](docs/T_slice.png)

### Porous Convection 3D MPI
![porous-convection-3d-mpi](docs/porous-3d-multixpu.gif)

