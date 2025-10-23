"""
Setup script for YOLO Hyperparameter Benchmarking System
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="yolo-benchmark",
    version="1.0.0",
    author="Miguel Batista Pinotti",
    author_email="miguel.pinotti@unip.br",
    description="Comprehensive YOLO hyperparameter benchmarking system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/miguel-pinotti/yolo-benchmark",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "yolo-benchmark=cli.main:main",
            "yolo-bench=cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="yolo, hyperparameter, benchmark, computer-vision, object-detection",
    project_urls={
        "Bug Reports": "https://github.com/miguel-pinotti/yolo-benchmark/issues",
        "Source": "https://github.com/miguel-pinotti/yolo-benchmark",
        "Documentation": "https://yolo-benchmark.readthedocs.io/",
    },
)
