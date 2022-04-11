from distutils.core import setup

requirements = [
        "numba",
        "numpy",
        "tqdm",
        "pandas",
        "matplotlib",
        "scipy",
        "sympy",
        ]

setup(
        name="MolLaserCoolSimulations",
        version="2.4.3",
        author="Felix Kogel",
        author_email="fkogel@pi5.physik.uni-stuttgart.de",
        description="A module for calculating molecular laser-cooling dynamics and molecular spectra",
        python_requires=">=3.6",
        install_requires=requirements
        )
