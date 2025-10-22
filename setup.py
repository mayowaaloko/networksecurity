"""
the setup.py file is an essential part of packaging and distributing
python projects. it is used by setuptools to define the configuration
of your project, such as its metadata, dependencies, and more.
"""

from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """
    this function will return the list of requirements
    """
    requirement_lst: List[str] = []

    try:
        with open("requirements.txt", "r") as file:
            # Read lines from file
            lines = file.readlines()
            ##process each line
            for line in lines:
                requirement = line.strip()

                ## ignore empty lines and -e .
                if requirement and not requirement.startswith("-e"):
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt not found.")
    return requirement_lst


print(get_requirements("requirements.txt"))


setup(
    name="networksecurity",
    version="0.0.1",
    author="Mayowa Aloko",
    author_email="mayowaaloko@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
