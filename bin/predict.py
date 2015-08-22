#!/usr/bin/env python
"""
Reads a TSV file from stdin and adds a column with
a models score on it. Assumes that there are two clusters.
"""
import sys

from argparse import ArgumentParser
from scipy.misc import logsumexp

from openmic.em import load_model
from openmic.feature import Features

def main(args):
    feature_map = Features.load(args.feature_file, args.models_dir)
    model = load_model(feature_map, "%s/%s/em.model" % (args.models_dir, args.model))
    for line in sys.stdin:
        x = line.strip().split('\t')
        score = predict(model, x)[0]
        print '\t'.join(x + [str(score)])

def predict(model, x):
    p0 = model.predict(0, x) + model.logprior(0)
    p1 = model.predict(1, x) + model.logprior(1)
    sum = logsumexp([p0, p1])
    return -p0 + sum, -p1 + sum

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--models_dir', type=str,
                  help='The directory where models are located',
                           default='models')
    argparser.add_argument('--feature_file', type=str,
                  help='The features JSON file',
                  default="features.json")
    argparser.add_argument('model', type=str,
                  help='The model to test')
    args = argparser.parse_args()
    main(args)
