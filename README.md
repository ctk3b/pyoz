## Iterative Ornstein-Zernike equation solver

[![Build Status](https://travis-ci.com/ctk3b/pyoz.svg?token=T2bs5CWLhhVcoq5EpoJT&branch=master)](https://travis-ci.com/ctk3b/pyoz)
[![BSD 3-Clause](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](license.md)

This is a continuation of the [pyOZ project](http://pyoz.vrbka.net) by Lubos Vrbka
which was discontinued in 2009 at version 0.3.0. 

### [Installation instructions](docs/installation.md)

### Getting started

Proper docs coming soon. In the meantime, check out the `examples` folder.

### General features
- [x] Bulk calculations for simple atomic/ionic systems
- [x] Simple iteration scheme (direct Picard iteration)
- [ ] Newton-Raphson/conjugate gradient iteration scheme

### Closure relations
- [x] Percus-Yevick (PY)
- [x] Hypernetted Chain (HNC)
- [x] Reference Hypernetted Chain (RHNC)

### Interatomic potentials
- [x] Continuous short-range potentials (e.g. Lennard-Jones, screened Coulomb)
- [ ] Discontinuous potentials
- [ ] Long-range potentials using Ng-renormalization (e.g. Coulomb)

### Thermodynamic properties
- [ ] Osmotic coefficient
- [ ] Isothermal compressibility
- [x] Excess chemical potential
- [x] Kirkwood-Buff factor

### Output
- [x] Pair correlation functions g(r)
- [x] Direct correlation functions c(r)
- [x] Partial structure factors S(k)
- [x] Pair potentials U(r)
