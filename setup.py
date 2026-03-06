from setuptools import setup, find_packages

setup(
    name="organoid_rl",
    version="0.1.0",
    description="OrganoidRL: Biologically-inspired reinforcement learning with spiking neural networks",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "brian2>=2.5.1",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
