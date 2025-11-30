"""
Setup script for FastArray - A compressed array library for AI models
"""
from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="FastArray",
    version="0.1.0",
    author="FastArray Development Team",
    author_email="fastarray@example.com",
    description="A compressed array library for AI models - drop-in replacement for NumPy with automatic compression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fastarray/fastarray",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.21.0",
        "blosc>=1.0.0; python_implementation != 'PyPy'",
        "scipy>=1.7.0; extra == 'scipy'",
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'mypy>=0.910',
            'black>=21.0',
            'flake8>=3.8',
        ],
        'scipy': [
            'scipy>=1.7.0',
        ],
    },
    keywords="numpy array compression machine-learning ai tensor",
    project_urls={
        "Bug Reports": "https://github.com/fastarray/fastarray/issues",
        "Source": "https://github.com/fastarray/fastarray",
        "Documentation": "https://fastarray.readthedocs.io/",
    }
)