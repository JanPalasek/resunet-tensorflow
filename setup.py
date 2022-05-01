import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resunet",
    version="1.1",
    package_dir={
        "": "src"
    },
    packages=setuptools.find_packages("src"),
    install_requires=["tensorflow>=2.0.0"],
    python_requires=">=3.6",
)