from setuptools import setup, find_packages
import os

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line and not line.startswith('#')]

setup(
    name="FreeAeon-Fractal",
    version="0.3.1",
    author="Jim Xie",
    author_email="jim.xie.cn@outlook.com",
    description="Toolkit for Multifractal Analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jim-xie-cn/FreeAeon-Fractal",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=parse_requirements('requirements.txt'),
    include_package_data=True,
    package_data={
        "": ["images/*", "demo.py"],
    },
)
