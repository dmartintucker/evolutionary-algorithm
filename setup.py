import setuptools

setuptools.setup(
    name = "genetic-algorithm",
    version = "0.0.1",
    author = "Daniel Tucker",
    author_email = "ryan.solgi@gmail.com",
    maintainer = "Daniel M Tucker",
    description = "A simple evolutionary algorithm implementation in Python",
    url="https://github.com/dmartintucker/genetic-algorithm",
    keywords=["Python", "evolutionary", "genetic", "algorithm", "GA"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=['func-timeout', 'numpy']
)