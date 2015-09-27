#!/usr/bin/env python
"""
# EXPERIMENTAL
Reads a TSV file from stdin and outputs a BED Graph file with
a models score on it. Assumes that there are two clusters.
"""
import sys

import tabix
from argparse import ArgumentParser
from scipy.misc import logsumexp

from sscm.em import load_model
from sscm.feature import Features

def main(args):
    feature_map = Features.load(args.feature_file)
    model = load_model(feature_map, "%s/%s/em.model" % (args.models_dir, args.model))
    tb = tabix.open(args.snp_file)
    print "track type=bedGraph name=%s" % args.model
    current_location = None
    current_score = (None, float('-inf'))
    for result in tb.querys(args.region):
        x = result + ["NA", "NA", "NA"]
        cm = x[0], x[1]

        score = predict(model, result)[0]
        if current_location is None:
            current_location = cm
            current_score = (x, score)
        else:
            if cm == current_location:
                if score > current_score[1]:
                    current_score = (x, score)
            else:
                print '\t'.join(['chr' + current_score[0][0], current_score[0][1], str(int(current_score[0][1]) + 1), str(current_score[1])])
                current_location = cm
                current_score = (x, score)
    print '\t'.join(['chr' + current_score[0][0], str(int(current_score[0][1]) - 1), current_score[0][1], str(current_score[1])])


def predict(model, x):
    p0 = model.predict(0, x) + model.prior(0)
    p1 = model.predict(1, x) + model.prior(1)
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
    argparser.add_argument('--snp_file', type=str,
                  default='annotated_snps.tsv.gz')
    argparser.add_argument('region', type=str,
                  help='The region of the genome to be annotated')
    args = argparser.parse_args()
    main(args)
