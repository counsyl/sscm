"""
Represents the instance of the graphical model where
features are emitted directly from the cluster assignment
"""
import numpy as np
from scipy.misc import logsumexp
from math import log

from em import EM


class Updater(object):
    """
    Stores intermediary calculations and values used
    to update a parameter's value during an iteration
    of EM.
    """

    def __init__(self, info):
        self.info = info
        self.reset()

    def reset(self):
        """
        Resets current values to 0 and data to empty
        for the beginning of next iteration of EM
        """
        raise NotImplementedError()

    def handle_point(self, x, i, tau, indicator):
        """
        Updates current intermediary values with
        a point from the dataset.
        """
        raise NotImplementedError()

    def calculate_params(self, tau, old=None):
        """
        Calculates parameter values from intermediary
        values
        """
        raise NotImplementedError()


class GaussianUpdater(Updater):
    """
    Stores intermediary values used to update
    Gaussian parameters in this particular
    graphical model
    """

    def reset(self):
        self.points = []
        self.indicator = []

    def handle_point(self, x, i, tau, indicator):
        if not x:
            self.points.append(0)
        else:
            self.points.append(x)
        self.indicator.append(int(indicator))

    def calculate_params(self, tau, old=None):
        points = np.array(self.points)
        tau_ind = tau * self.indicator
        numer = np.dot(tau_ind, points)
        denom = np.sum(tau_ind)
        mu = numer / denom

        sigma = np.dot(tau_ind, (points - mu) ** 2) / denom
        assert sigma > 0
        return {
            'mu': mu,
            'sigma': sigma
        }


class MultivariateGaussianUpdater(Updater):
    """
    Used to update Multivariate Gaussian parameters in this particular
    graphical model
    """
    def __init__(self, info):
        self.names = info['names']
        super(MultivariateGaussianUpdater, self).__init__(info)

    def reset(self):
        self.sum = {}
        self.sum_count = {}
        self.cosum = {}
        self.cosum2 = {}
        self.cosum_count = {}
        for name in self.names:
            self.sum[name] = 0
            self.sum_count[name] = 0
            self.cosum[name] = {}
            self.cosum2[name] = {}
            self.cosum_count[name] = {}
            for coname in self.names:
                self.cosum[name][coname] = 0
                self.cosum2[name][coname] = [0, 0]
                self.cosum_count[name][coname] = 0

    def handle_point(self, x, i, tau, indicator):
        if indicator:
            items = x.items()
            for name, value in items:
                self.sum[name] += value * tau
                self.sum_count[name] += tau
                for coname, covalue in items:
                    self.cosum[name][coname] += tau * value * covalue
                    self.cosum2[name][coname][0] += tau * value
                    self.cosum2[name][coname][1] += tau * covalue
                    self.cosum_count[name][coname] += tau

    def calculate_params(self, tau, old=None):
        names = self.info['names']
        mu = {}
        sigma = {}
        for name in names:
            if self.sum_count[name] == 0:
                mu[name] = 0
                continue
            mu[name] = self.sum[name] / self.sum_count[name]

        for name in names:
            sigma[name] = {}
            for coname in names:
                if self.cosum_count[name][coname] != 0:
                    sigma[name][coname] = (1/self.cosum_count[name][coname] *
                                           (self.cosum[name][coname] -
                                            self.cosum2[name][coname][0] *
                                            mu[coname] -
                                            self.cosum2[name][coname][1] *
                                            mu[name] +
                                            self.cosum_count[name][coname] *
                                            mu[name] * mu[coname]))
                else:
                    sigma[name][coname] = old['sigma'][name][coname]

        return {
            'mu': mu,
            'sigma': sigma
        }


class MultinomialUpdater(Updater):
    """
    Used to update Multinomial parameters in this particular
    graphical model
    """
    def __init__(self, info):
        self.categories = info['categories']
        super(MultinomialUpdater, self).__init__(info)

    def handle_point(self, x, i, tau, indicator):
        if indicator:
            self.counts[x] += tau

    def calculate_params(self, tau, old=None):
        params = {}
        total = float(sum(self.counts.itervalues()))
        for category, tau_sum in self.counts.iteritems():
            params[category] = tau_sum / total
            assert params[category] >= 0 and params[category] <= 1
        return params

    def reset(self):
        self.counts = {c: 0 for c in self.categories}

UPDATER_MAP = {
    "Gaussian": GaussianUpdater,
    "Multinomial": MultinomialUpdater,
    "MultivariateGaussian": MultivariateGaussianUpdater
}


class IndependentGenerativeModel(EM):
    """
    Stores the parameters and computes likelihood
    based on a Naive Bayes generative model for clustering.
    pi is the prior distribution on clusters and
    theta is the parameter matrix for the clusters.
    """

    name = "IndependentGenerativeModel"

    def init_parameters(self):
        """
        Initializes parameters for this particular graphical model
        """
        parameters = {}
        parameters["pi"] = np.array([0.999] +
                                    [0.001 / (self.K - 1)] * (self.K - 1))
        parameters["theta"] = {}
        for k in xrange(self.K):
            parameters["theta"][k] = {}
            for j in xrange(self.num_features):
                feature = self.features[j]
                updater = UPDATER_MAP[feature.distribution]
                parameters["theta"][k][j] = feature.parameter_type(
                    updater, feature.info)
                parameters["theta"][k][j].init_parameter(feature.info)
        self.parameters = parameters

    def compute_loglikelihood(self, k, x):
        """
        Calculates the log-likelihood of a row being generated
        by the graphical model for a particular cluster
        """
        parameters = self.parameters
        log_likelihoods = [
            parameters['theta'][k][j].logpdf(self.features[j].get(x))
            for j in xrange(self.num_features)
            if self.features[j].indicator(x)]

        total_log_likelihood = log(parameters["pi"][k]) \
            + sum(log_likelihoods)
        return total_log_likelihood

    def compute_log_tau(self, x):
        """
        Calculates the log-likelihood of a row being generated
        by the graphical model for each different cluster.
        """
        log_tau = np.zeros(self.K)
        log_tau += np.array([self.compute_loglikelihood(k, x)
                            for k in xrange(self.K)])
        log_tau -= logsumexp(log_tau)
        return log_tau

    def get_parameters(self):
        parameters = self.parameters["theta"]
        params = []
        params.extend(self.parameters["pi"].tolist())
        for k in xrange(self.K):
            for j in xrange(self.num_features):
                params.extend(parameters[k][j].value.values())
        return np.array(params)

    def get_parameters_as_dict(self):
        params = {}
        params["pi"] = self.parameters["pi"].tolist()
        params["alpha"] = self.alpha
        params["theta"] = {}
        for k in xrange(self.K):
            params["theta"][k] = {}
            for j in xrange(self.num_features):
                params["theta"][k][j] = \
                    self.parameters["theta"][k][j].get_all()
        return params

    def load_parameters_from_dict(self, init):
        """
        Used when loading a model from a dump file
        """
        self.alpha = init["parameters"]["alpha"]
        self.parameters["pi"] = np.array(init["parameters"]["pi"])
        for k in xrange(self.K):
            for j in xrange(self.num_features):
                self.parameters["theta"][k][j].set_all(
                    init["parameters"]["theta"][k][j])

    def fit(self, k, data_fp):
        """
        Performs MLE for a dataset for cluster k
        """
        parameters = self.parameters
        count = 0
        for line in data_fp:
            x = line.strip().split('\t')
            count += 1
            for j in xrange(self.num_features):
                feature = self.features[j]
                if feature.indicator(x):
                    parameters["theta"][k][j].add_sample(feature.get(x))
        self.N_fit = count
        for j in xrange(self.num_features):
            parameters["theta"][k][j].fit()

    def parameter_iterator(self):
        """
        An iterator over all the parameters used by EM
        """
        for k in xrange(self.K):
            for j in xrange(self.num_features):
                    yield k, self.features[j], self.parameters["theta"][k][j]

    def print_parameters(self):
        print "Mixture Weights:", self.parameters["pi"]
        for j in xrange(self.num_features):
            for k in xrange(self.K):
                    print "%s[%u]" % (self.features[j].name, k),
                    print self.parameters["theta"][k][j].get_all()
