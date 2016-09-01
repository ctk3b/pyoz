### Installation instructions


You must currently build ``pyoz`` from source using **Python 3.4+**.
In the near future, ``pip`` and ``conda`` packages will become available.

Until then, we recommend that you install the ``pyoz`` dependencies using
 [``conda``](https://www.continuum.io/downloads):
 
```bash
conda install scipy pandas numba
pip install -e git+https://github.com/ctk3b/pyoz.git#egg=pyoz
cd src/pyoz
```

but, in theory, you can also install them via pip:

```bash
pip install scipy pandas numba
pip install -e git+https://github.com/ctk3b/pyoz.git#egg=pyoz
cd src/pyoz
```

#### Testing your installation

The test suite uses ``pytest`` and the ``hypothesis`` plugin. You can install
both of these packages via ``pip``:

```bash
pip install pytest hypothesis
```

and then run the tests using

```bash
py.test -v
```