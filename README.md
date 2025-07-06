# 1D Quantum Simulation: Numerical Solutions to the Schrödinger Equation

## Overview
This project implements numerical methods to solve the one-dimensional time-independent Schrödinger equation using Python. It explores various computational techniques, including finite difference methods and Numerov's method, to compute eigenvalues and eigenfunctions for different quantum potential wells. 

Simulations include:
- Harmonic oscillator  
- Finite well  
- Potential barrier  
- Double well  
- Radial part of the hydrogen atom

The project focuses on visualizing quantum phenomena such as confinement, tunneling, and level splitting.

---

## Project Structure

### Main Code
- **Main Oscillator.py**: Main script for running simulations, especially for the harmonic oscillator.  
- **Numerov.py**: Implementation of Numerov's method for solving the Schrödinger equation.

### Images and GIFs
- `Barrier/` : Visualizations for the potential barrier simulation.  
- `Double well/` : Visualizations for the double well potential, showcasing tunneling effects.  
- `Finite well/` : Visualizations for the finite potential well.  
- `Harmonic oscillator/` : Visualizations for the harmonic oscillator potential.  
- `Hydrogen/` : Visualizations for the radial solutions of the hydrogen atom.  
- `Numerov/` : Images related to Numerov's method results.  
- `Numerical errors/` : Images showing numerical error analysis for each simulation.  
- `Length errors/` : Images analyzing errors related to spatial grid size.

### Documentation and Misc
- **Quantum-Simulation-MainText-FR.pdf**: Comprehensive documentation in French detailing theory, numerical methods, and results, including visualizations.  
- **README.md**: This file.  
- **LICENSE**: Project license information.

---

## Objectives

Numerically solve the 1D time-independent Schrödinger equation for:

- **Harmonic Oscillator**: Particle under quadratic potential, with analytical solutions for validation.  
- **Finite Well**: Particle in a confined finite-depth potential.  
- **Potential Barrier**: Analysis of quantum tunneling and scattering states.  
- **Double Well**: Level splitting due to tunneling effects.  
- **Hydrogen Atom (Radial)**: Radial part of the 3D Schrödinger equation for hydrogen-like orbitals.

The project demonstrates the effectiveness of numerical methods (finite differences and Numerov's), achieving relative errors between `10^-3` and `10^-7` for grid sizes around `N ~ 1000`. Analytical solutions for hydrogen complement the numerical results.

---

## Key Features

### Numerical Methods
- Finite difference discretization of the Schrödinger equation  
- Numerov's method for high-accuracy solutions of second-order differential equations  
- Matrix diagonalization to find eigenvalues and eigenfunctions

### Visualizations
- Plots of potentials, energy levels, and probability densities  
- Comparison of numerical and analytical solutions (where available)  
- Numerical error and convergence analysis as functions of grid size (\(N\)) and domain length (\(L\))

### Quantum Phenomena
- Visualization of quantum confinement, tunneling, and level splitting  
- Symmetric and antisymmetric wavefunctions in double wells  
- Radial wavefunctions and energy levels for hydrogen atom, illustrating Bohr radius and orbital localization

---

## Dependencies

- Python 3.x  
- Libraries: [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/)  
- Optional: LaTeX (for rendering equations in documentation)

---

## Usage

### Setup
1. Ensure Python 3.x and required libraries are installed:
   ```bash
   pip install numpy scipy matplotlib
