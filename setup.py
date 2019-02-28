#!/usr/bin/env python3

from setuptools import (
        setup,
        find_packages,
        )

VERSION = '0.0.6.2'

setup(
    name='cherry-rl',
    packages=find_packages(),
    version=VERSION,
    description='PyTorch Reinforcement Learning Framework for Researchers',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Seb Arnold',
    author_email='smr.arnold@gmail.com',
    url = 'https://seba-1511.github.com/cherry',
    download_url = 'https://github.com/seba-1511/cherry/archive/' + str(VERSION) + '.zip',
    license='License :: OSI Approved :: Apache Software License',
    classifiers=[],
    scripts=[],
    install_requires=[
        'numpy>=1.15.4',
        'gym>=0.10.9',
        'torch>=1.0.0',
    ],
)
