from distutils.core import setup

requirements = [
        "numba",
        "numpy==1.21.1",
        "tqdm",
        "pandas",
        "matplotlib",
        "scipy",
        "sympy",
        ]

setup(
        name="Great Simulation",
        version="2.4.3",
        author="Felix Kogel",
        install_requires=requirements
        )
