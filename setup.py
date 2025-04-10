"""
Setup file for the tf-model-analyzer package
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tf-model-analyzer",
    version="0.1.0",
    author="lognjen",
    author_email="ognjen.maksimovic@gmail.com",
    description="A CLI tool for analyzing TensorFlow models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lognjen/tensorflow-model-analyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "tf-analyzer=tf_model_analyzer.cli:main",
        ],
    },
)
