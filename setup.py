from setuptools import setup, find_packages

setup(
    name="stplfcnn",
    version="1.0",
    description=(
        "Short-Term Probabilistic Load Forecasting at Low Aggregation Levels"
        " using Convolutional Neural Networks"
    ),
    author="Alexander Elvers",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=0.23,<0.24",
        "numpy>=1.14",
        "scipy>=1.1",
        "matplotlib>=3",
        "scikit-learn",
        "patsy",
        "dataclasses; python_version<'3.7'",
        "PyYAML>=3.12,<4",
        "hyperopt==0.1.1",
        "click>=6,<7",
    ],
    extras_require=dict(
        tfcpu=["tensorflow>=1.8,<1.9"],
        tfgpu=["tensorflow-gpu>=1.8,<1.9"],
        dev=["pytest", "pytest-mock"],
    ),
    entry_points=dict(console_scripts=["stplfcnn = stplfcnn.cli:cli"]),
)
