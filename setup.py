from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="coupling",
    version="0.1",
    author="Murdock Aubry, Haoming Meng, Anton Sugolov, Vardan Papyan",
    author_email="asugolov@gmail.com",
    description="A package for computing coupling from PyTorch hooks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sugolov/coupling",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)