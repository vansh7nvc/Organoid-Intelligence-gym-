from setuptools import setup, find_packages

setup(
    name="organoid_rl",
    version="1.0.0",
    description="OrganoidEnv: A Stabilized Reinforcement Learning Environment for Biological Spiking Networks",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Vansh Sharma",
    author_email="sharmavansh1409@gmail.com",
    license="MIT",
    url="https://github.com/vansh-sharma/organoid-rl",
    project_urls={
        "Bug Tracker": "https://github.com/vansh-sharma/organoid-rl/issues",
        "Documentation": "https://github.com/vansh-sharma/organoid-rl#readme",
    },
    packages=find_packages(),
    install_requires=[
        "brian2>=2.5.1",
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
