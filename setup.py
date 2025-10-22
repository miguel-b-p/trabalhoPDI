"""
Setup script para YOLO Benchmark System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Lê README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="yolo-benchmark-system",
    version="1.0.0",
    author="YOLO Benchmark Team",
    description="Sistema completo de benchmark e testes de parâmetros YOLO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yolo-benchmark-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "bokeh>=3.0.0",
        "jupyter_bokeh>=4.0.0",
        "ipywidgets>=8.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "jupyter": [
            "jupyterlab>=4.0.0",
            "notebook>=7.0.0",
            "ipython>=8.0.0",
        ],
    },
)
