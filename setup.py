"""Setup script for MANTIS"""
from setuptools import setup, find_packages

setup(
    name="MANTIS",
    version="0.1.0",
    packages=find_packages(exclude=["logs*", "data*", "models*", "venv*", "test_storage*", "*.tests*", "*.test*"]),
    include_package_data=True,
    package_data={
        "*": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)
