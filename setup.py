#!/usr/bin/env python

from setuptools import setup


def main():
    setup(
        name='sscm',
        version='1.0',
        description=('Semi-Supervised Clustering Method for variant '
                     'pathogenicity prediction'),
        author='Sharad Vikram, Matt Rasmussen',
        author_email='rasmus@counsyl.com',
        packages=['sscm'],
        scripts=[
            'bin/sscm-predict',
            'bin/sscm-track',
            'bin/sscm-train',
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
