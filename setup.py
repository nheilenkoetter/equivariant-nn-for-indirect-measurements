import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "equivariant-nn-for-indirect-measurements",
    version = "0.0.1",
    author = "Nick Heilenkötter",
    author_email = "heilenkoetter@uni-bremen.de",
    description = ("An demonstration of how to create, document, and publish "
                                   "to the cheese shop a5 pypi.org."),
    license = "Apache License 2.0",
    keywords = "inverse problems equivariant deep learning",
    url = "https://github.com/nheilenkoetter/equivariant-nn-for-indirect-measurements",
    packages=['equivariant_nn_for_indirect_measurements'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Equivariant Neural Networks",
        "License :: OSI Approved :: Apache License 2.0",
    ],
)