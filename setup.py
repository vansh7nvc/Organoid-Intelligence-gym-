"""
Setup script for OrganoidRL / OrganoidEnv.
Supports backward-compatible installation via pip and setuptools.
"""

from setuptools import setup, find_packages
import os

readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = open(readme_path, encoding="utf-8").read() if os.path.exists(readme_path) else ""

setup(
    name="organoid_rl",
    version="1.0.0",
    description="OrganoidEnv: A Stabilized Reinforcement Learning Environment for Biological Spiking Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vansh Sharma, Dr. Seema Malik",
    author_email="sharmavansh1409@gmail.com",
    license="MIT",
    url="https://github.com/vansh7nvc/Organoid-Intelligence-gym-",
    project_urls={
        "Bug Tracker": "https://github.com/vansh7nvc/Organoid-Intelligence-gym-/issues",
        "Documentation": "https://github.com/vansh7nvc/Organoid-Intelligence-gym-#readme",
        "Source Code": "https://github.com/vansh7nvc/Organoid-Intelligence-gym-",
    },
    packages=find_packages(include=["organoid_rl", "organoid_rl.*"]),
    python_requires=">=3.9",
    install_requires=[
        "brian2>=2.8.0",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "torch>=2.0.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "ruff>=0.1.0",
            "black>=23.0.0",
        ],
        "baselines": [
            "stable-baselines3>=2.0.0",
            "ray[tune]>=2.9.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
