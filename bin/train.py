#!/usr/bin/env python
"""
The script used to actually train models
"""
import sys
import os
import os.path as path
import signal
import argparse
import shutil

from openmic.feature import Features
from openmic.model import IndependentGenerativeModel
from openmic.em import load_model, save_model


def train_model(name, features, train, feature_file,
                fit=None, num_clusters=2,
                alpha=0.01, threshold=0.000001,
                models_dir='models', replace=False,
                load=None, mle=False,
                ):

    # loading feature information from JSON file
    feature_map = Features.load(feature_file, models_dir)

    with open(train, 'r') as train_fp:
        train_fp = open(train, 'r')
        em = IndependentGenerativeModel(feature_map, K=num_clusters,
                                        alpha=alpha, threshold=threshold)

        print "Features:", features
        em.add_features(features)

        if fit:
            print 'Fitting benign cluster...'
            with open(fit, 'r') as fit_fp:
                em.fit(0, fit_fp)
                n0 = em.N_fit
            em.hold_cluster(0)

        if mle:
            print 'Fitting pathogenic cluster...'
            with open(train, 'r') as train_fp:
                em.fit(1, train_fp)
                n1 = em.N_fit
            em.parameters["pi"][0] = float(n0) / (n0 + n1)
            em.parameters["pi"][1] = 1 - em.parameters["pi"][0]

        if load is not None:
            em = load_model(feature_map, '%s/%s/em.model' % (models_dir, args.load))
            print "Initial Parameters:", em.parameters

        def end(success):
            print "Saving model..."
            save_model(em, "%s/%s/em.model" % (args.models_dir, args.name))

        def sigterm_handler(signum, frame):
            print 'Signal handler called with signal', signum
            end(True)
            sys.exit(0)

        signal.signal(signal.SIGINT, sigterm_handler)

        if not mle:
            em.run(train_fp)
        end(True)


def main(args):
    name, models_dir = args.name, args.models_dir

    if not path.isdir('%s/' % models_dir):
        os.mkdir('%s/' % models_dir)

    if args.replace:
        if path.isdir('%s/%s' % (models_dir, name)):
            shutil.rmtree('%s/%s' % (models_dir, name))

    if not args.load:
        if path.isdir('%s/%s' % (models_dir, name)):
            raise IOError("Model already exists")
        os.mkdir('%s/%s' % (models_dir, name))

    train_model(args.name, args.features, args.train, args.feature_file,
        fit=args.fit,
        num_clusters=args.num_clusters,
        alpha=args.alpha,
        threshold=args.threshold,
        models_dir=args.models_dir,
        replace=args.replace,
        load=args.load,
        mle=args.mle)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('name', type=str,
                           help='The name of the model to be trained')
    argparser.add_argument('--feature_file', type=str,
                           help='The feature configuration JSON file',
                           default='features.json')
    argparser.add_argument('--features', nargs='+', required=True, type=str,
                           help='The features to be used in the model')
    argparser.add_argument('--train', required=True, type=str,
                           help='The training file')
    argparser.add_argument('--fit', type=str,
                           help='the file to fit the benign cluster')
    argparser.add_argument('--num_clusters', default=2, type=int,
                           help='The number of clusters to fit')
    argparser.add_argument('--alpha', default=0.01, type=float,
                           help='The dirichlet parameter')
    argparser.add_argument('--threshold', default=0.000001, type=float,
                           help='The difference at which to stop iterating')
    argparser.add_argument('--replace', action='store_true',
                           help='Replace the model with the same name if it exists')
    argparser.add_argument('--models_dir', default='models', type=str,
                           help='Set the directory in which to save the model')
    argparser.add_argument('--load', type=str,
                           help='A model to load from')
    argparser.add_argument('--mle', action='store_true', help='Directly fit pathogenic cluster with MLE')

    args = argparser.parse_args()
    main(args)
