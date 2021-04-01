from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pymul",
    version="0.0.1",
    author="mathletedev",
    author_email="mathletedev@gmail.com",
    url="https://github.com/mathletedev/pymul",
    license="Apache License 2.0",
    description="A Python machine learning library!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["pymul", "pymul/functions", "pymul/layers"],
    install_requires=["numpy >= 1.20.2"],
)
