import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evolutionary_algorithm",
    version="0.0.1",
    author="Daniel Tucker",
    author_email="dmartintucker@gmail.com",
    maintainer="Daniel Tucker",
    description="An evolutionary (genetic) algorithm designed specifically for optimizing predictive models with integer, real, boolean, and categorical inputs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dmartintucker/evolutionary-algorithm",
    keywords=["Python", "evolutionary", "genetic", "algorithm", "GA"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires='>3.6',
    install_requires=['func-timeout', 'numpy']
)
