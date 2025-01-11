from setuptools import setup, find_packages

setup(
    name="datatools",
    version="0.1",
    description="some tools",
    author="marcu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "plotly",
        "polars",
        "regex",
    ],
    scripts=[
        "sh/clearoutput.sh",
    ],
)
