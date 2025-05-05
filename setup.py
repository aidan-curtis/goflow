"""Setup script."""

from setuptools import setup

setup(
    name="goflow",
    version="0.1.0",
    packages=["goflow"],
    include_package_data=True,
    install_requires=[
        "pybullet",
        "numpy",
        "torch",
        "scipy",
        "zuko"
    ],
)