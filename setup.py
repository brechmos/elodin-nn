#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- encoding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='transfer_learning',
    version='0.2.dev0',
    license='GPLV3.0',
    description='Transfer learning info.',
    long_description='Transfer Learning',
    author='Craig Jones',
    author_email='craig@brechmos.org',
    url='https://github.com/brechmos-stsci/transfer-learning',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Framework :: Pytest',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Software Development :: Testing',
        'Topic :: Utilities',
    ],
    keywords=[ 'detect', 'open', 'file', 'handle', 'psutil', 'pytest', 'py.test' ],
    install_requires=[ 'pytest>=2.8.0', 'psutil', 'keras', 'matplotlib', 'tensorflow', 'imageio', 'requests', 'blitzdb', 'celery', 'scikit-learn', 'scikit-image', 'redis', 'jupyter', 'ipython' ],
    python_requires='>=3.5'
)
