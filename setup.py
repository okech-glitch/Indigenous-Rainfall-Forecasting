from setuptools import setup, find_packages

setup(
    name="indigenous-rainfall-forecasting",
    version="0.1.0",
    description="Rainfall type prediction using Indigenous Ecological Indicators",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[],
)
