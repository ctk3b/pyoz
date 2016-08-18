## Iterative Ornstein-Zernike equation solver

[![Build Status](https://travis-ci.com/ctk3b/pyoz.svg?token=T2bs5CWLhhVcoq5EpoJT&branch=master)](https://travis-ci.com/ctk3b/pyoz)

This is a continuation of the [pyOZ project](pyoz.vrbka.net) by Lubos Vrbka
which was discontinued in 2009 at version 0.3.0. 

### General features
- [x] Bulk calculations for simple atomic/ionic systems
- [x] Simple iteration scheme (direct Picard iteration)
- [ ] Newton-Raphson/conjugate gradient iteration scheme

### Closure relations
- [x] Hypernetted Chain (HNC)
- [x] Percus-Yevick (PY)

### Interatomic potentials
- [ ] Hard spheres potential
- [ ] Coulomb potential (used together with Ng-renormalization)
- [ ] Charge-induced dipole interaction
- [x] Lennard-Jones potential with σ and ε
- [ ] Potential of mean force from external file

### Thermodynamic properties
- [ ] Osmotic coefficient
- [ ] Isothermal compressibility
- [ ] Excess chemical potential
- [x] Kirkwood-Buff factor

### Output
- [x] Pair correlation functions g(r)
- [x] Direct correlation functions c(r)
- [ ] Partial structure factors S(k)
- [x] Pair potentials U(r)
- [x] Total interaction potential - U(r) + indirect correlation function
