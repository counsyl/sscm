#!/usr/bin/env python

from setuptools import setup


def main():
    setup(
        name='sscm',
        version='1.0',
        description=('Semi-Surpervised Clustering Method for variant '
                     'pathogenicity prediction'),
        author='Sharad Vikram, Matt Rasmussen',
        author_email='rasmus@counsyl.com',
        packages=['sscm'],
        include_package_data=True,
        package_data={
            '': ['sscm/data/*'],
        },
        scripts=[
            'bin/predict.py',
            'bin/track.py',
            'bin/train.py',
        ],
        install_requires=[
            'argparse',
            'numpy',
            'progressbar',
            'pytabix',
            'scipy',
            'wsgiref',
        ],
    )

if __name__ == '__main__':
    main()
