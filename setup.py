"""Packaging for svdfusion."""

from setuptools import find_packages, setup

setup(
    name="svdfusion",
    version="1.0.0",
    description="SVD-Fusion CAM: Hierarchical Robust Visual Explanations "
                "via Singular Value Decomposition",
    author="Newsha Shahbodaghkhan",
    packages=find_packages(exclude=("scripts", "configs")),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "opencv-python>=4.5",
        "scipy>=1.7",
        "matplotlib>=3.4",
        "pyyaml>=5.4",
        "tqdm>=4.60",
    ],
    extras_require={"tf": ["tensorflow>=2.9"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
    ],
)
