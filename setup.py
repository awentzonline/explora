import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="explora",
    version="0.0.1",
    author="Adam Wentz",
    author_email="adam@adamwentz.com",
    description="Evolutionary apporaches to LoRA",
    long_description=read("README.md"),
    license="MIT",
    url="https://github.com/awentzonline/explora",
    packages=find_packages(),
    install_requires=[
        'click',
        'datasets',
        'lightning',
        'peft',
        'numpy',
        'ray',
        'transformers',
        'torch',
    ]
)
