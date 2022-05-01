from setuptools import setup, find_packages

setup(
    name="resunet",
    version="1.1a",
    packages=find_packages(),
    install_requires=["tensorflow>=2.0.0"],
    python_requires=">=3.6",
)