# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

version = '0.2'
name = 'tweet2vec'
short_description = 'tweet2vec model for Japanese language text'
author = 'Kensuke Mitsuzawa'
author_email = 'kensuke.mit@gmail.com'
url = ''
license = 'MIT'
classifiers = [
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Natural Language :: Japanese",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5"
        ]

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()


install_requires = ['typing', 'six', 'lasagne', 'Theano']
dependency_links = []

setup(
    name=name,
    version=version,
    description=short_description,
    long_description=long_description,
    author=author,
    author_email=author_email,
    install_requires=install_requires,
    dependency_links=dependency_links,
    url=url,
    license=license,
    packages=find_packages(),
    classifiers=classifiers,
    test_suite='tests',
    include_package_data=True,
    zip_safe=False
)