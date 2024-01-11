from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        return requirements

setup(
    name='src',
    author='C.C-KEH',
    version='0.0.1',
    author_email="email@gmail.com",
    install_requires=get_requirements('requirements.txt'),
    packages = find_packages()
)