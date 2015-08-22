"""
Performs EM on a tab separated dataset
"""
import numpy as np
import progressbar as pb
import sys


class EM(object):

    def __init__(self, feature_map, K=2, alpha=0.1, threshold=0.0001, iteration_limit=10000):
        """
        feature_map: generated from feature.py, which stores a mapping
                     between feature name and column index in the tsv
        K: number of clusters
        alpha: hyperparameter for Dirichlet prior
        threshold: value (L2 difference of parameters) at
                   which to stop EM iterations
        """
        self.features = []
        self.fm = feature_map
        self.K = K
        self.alpha = alpha
        self.hold = set()  # set of clusters which we don't update
        self.threshold = threshold
        self.iteration_limit = iteration_limit

    @property
    def num_features(self):
        return len(self.features)

    def add_feature(self, feature):
        self.add_features([feature])

    def add_features(self, features):
        self.features.extend(map(self.fm.get_feature, features))
        self.init_parameters()

    def run(self, data_fp):
        """
        Main method that EM is run from

        data_fp: an open file object that points to a TSV file
                 containing the data that EM will be run on
        """
        data = [line.strip().split('\t') for line in data_fp]
        N = len(data)
        self.N = N
        assert N > 0, "Can't train with empty dataset"

        self.iteration = 0

        # keeping track of the differences in parameters between iterations
        old_params, current_params = None, self.get_as_vector()
        param_diff = float('inf')

        # allocating the N x K matrix to hold
        # posterior probabilities of belonging
        # to a cluster
        tau = np.zeros((N, self.K))

        # looping until parameters converge
        while param_diff > self.threshold:

            # clears parameters of their accumulated data
            # from the previous iteration
            for k, feature, parameter in self.parameter_iterator():
                parameter.reset_sums()

            # logging
            progress_bar = pb.ProgressBar(maxval=N)
            progress_bar.start()
            if self.iteration > self.iteration_limit:
                raise Exception("too many iterations: %u"% self.iteration)

            self.iteration += 1

            print >>sys.stderr, "Iteration %u" % self.iteration
            self.print_parameters()

            # E step

            # looping through dataset
            for i in xrange(N):
                vec = data[i]

                # calculates log posterior probabilities
                # of belonging to all the clusters
                # and sets tau matrix
                log_tau = self.compute_log_tau(vec)
                tau[i, :] = np.exp(log_tau)

                # hands off tau values to each of the
                # parameters for updating later
                for k, feature, parameter in self.parameter_iterator():
                    if feature.indicator(vec):
                        parameter.handle_point(feature.get(vec),
                                               i,
                                               tau[i, k],
                                               True)
                    else:
                        parameter.handle_point(None, i, tau[i, k], False)
                progress_bar.update(i)
            progress_bar.finish()

            # M step

            # updating mixture weights
            self.parameters["pi"] = ((np.sum(tau, axis=0) +
                                     self.alpha / self.K - 1) /
                                     (N - self.K + self.alpha))

            # updating feature parameters
            for k, feature, parameter in self.parameter_iterator():
                if k not in self.hold:  # ignores clusters in hold
                    parameter.update(tau[:, k].T)

            # updating parameter difference
            old_params, current_params = current_params, \
                self.get_as_vector()
            param_diff = np.linalg.norm(current_params - old_params)
            print "Difference: %f" % param_diff

    def get_as_vector(self):
        params = []
        for k, feature, parameter in self.parameter_iterator():
            params.extend(parameter.get_as_vector())
        return np.array(params)

    def predict(self, k, x):
        """
        Calculates the likelihood of row vector x belonging to cluster k
        """
        likelihood = self.compute_loglikelihood(k, x)
        return likelihood

    def logprior(self, k):
        """
        Calculates the prior of cluster k
        """
        return np.log(self.parameters["pi"][k])

    def hold_cluster(self, k):
        """
        Tells the EM to not update cluster k
        """
        self.hold.add(k)


def save_model(model, location):
    """
    Dumps an EM model file to a location
    """
    model_info = {}
    model_info['name'] = model.name
    model_info['K'] = model.K
    parameters = model.get_parameters_as_dict()
    features = map(lambda x: x.name, model.features)
    model_info["parameters"] = parameters
    model_info["features"] = features
    # TODO: change to JSON serialization
    with open(location, 'w') as fp:
        fp.write(str(model_info))


def load_model(feature_map, location):
    """
    Given a feature_map loaded from feature.py and the location
    of a dumped EM model, will load them and return an EM model object
    """
    with open(location) as fp:
        # TODO: change to JSON deserialization
        model_info = eval(fp.read())
    from model import IndependentGenerativeModel
    model = IndependentGenerativeModel(feature_map, K=model_info["K"])
    model.add_features(model_info["features"])
    model.load_parameters_from_dict(model_info)
    return model
