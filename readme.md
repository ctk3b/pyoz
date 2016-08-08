## Iterative Ornstein-Zernike equation solver

### General features
* Bulk calculations for simple atomic/ionic systems
    * No atomic details in molecules and molecular ions on McMillan-Mayer level
* Simple iteration scheme (direct Picard iteration)
* Newton-Raphson/conjugate gradient iteration scheme

### Closure relations
* Hypernetted Chain (HNC)
* Percus-Yevick (PY)

### Interatomic potentials
* Hard spheres potential
* Coulomb potential (used together with Ng-renormalization)
* Charge-induced dipole interaction
* Lennard-Jones potential with σ and ε
* Potential of mean force from external file

### Thermodynamic properties
* Osmotic coefficient
* Isothermal compressibility
* Excess chemical potential
* Kirkwood-Buff factor

### Output
* Pair correlation functions g(r)
* Direct correlation functions c(r)
* Partial structure factors S(k)
* Pair potentials U(r)
* Total interaction potential - U(r) + indirect correlation functiol
