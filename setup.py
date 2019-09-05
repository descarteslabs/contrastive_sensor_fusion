"""
Run `pip3 install .` in the contrastive_sensor_fusion directory to install this project
as a module.
"""

from setuptools import find_packages, setup

setup(
    name="contrastive_sensor_fusion",
    description="Code for unsupervised learning by contrastive sensor fusion",
    version="0.0.1",
    packages=find_packages(),
    setup_requires=["setuptools>40"],
    install_requires=[],
)
