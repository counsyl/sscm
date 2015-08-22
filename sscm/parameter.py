"""
Contains the classes used as parameters and features in the graphical model
"""
import numpy as np
from numpy.random import normal, dirichlet
from math import log, pi, sqrt
from scipy.misc import logsumexp

PREPROCESS = {
    'log': lambda x: log(0.01 + x),
    'logit': lambda x: log((0.01 + x) / (1.01 - x)),
    'identity': lambda x: x
}


class Feature(object):
    """
    Represents a scalar feature in the graphical model and
    extracts said feature from a given data point (row
    in the dataset)
    """
    def __init__(self, name, column, feature_type, distribution,
                 preprocess=None, info=None):
        self.name = name
        self.column = column
        self.type = feature_type
        self.distribution = distribution
        self.parameter_type = FEATURE_TYPE[distribution]
        self.preprocess = (PREPROCESS[preprocess]
                           if preprocess
                           else PREPROCESS['identity'])
        self.info = info if info else {}

    def get(self, vector):
        """
        Given an row in the dataset, extracts this particular
        feature from the row, converting it to correct type in
        the process and also applying some pre-processing
        """
        return self.preprocess(self.type(vector[self.column]))

    def indicator(self, vector):
        """
        Returns whether true if the feature is not NA
        for this particular feature vector. Should
        be used before calling `feature.get()`
        """
        return vector[self.column].strip() != "NA"


class ModelFeature(Feature):
    """
    # EXPERIMENTAL
    Represents a nested model treated as an individual feature
    """
    def __init__(self, name, type, distribution, model_name,
                 models_dir, source_cluster=0, preprocess=None, info={}):
        super(ModelFeature, self).__init__(name, None, type,
                                           distribution,
                                           preprocess=preprocess, info=info)
        self.models_dir = models_dir
        self.source = source_cluster
        self.model_name = model_name
        self.cache = {}

    def load(self, fm):
        from em import load_model
        self.model = load_model(fm, "%s/%s/em.model" % (self.models_dir,
                                                        self.model_name))

    def get(self, vector):
        """
        Given an row in the dataset, extracts this particular feature from the row, converting
        it to correct type in the process and also applying some pre-processing
        """
        vector = tuple(vector)
        if vector in self.cache:
            return self.cache[vector]
        p0 = self.model.predict(1 - self.source, vector)
        p1 = self.model.predict(self.source, vector)
        log_sum = logsumexp([p0, p1])
        val = self.preprocess(-p0 + log_sum)
        self.cache[vector] = val
        return val

    def indicator(self, vector):
        """
        Returns whether that feature is NA or not for this particular feature vector. Should
        be used before calling `feature.get()`
        """
        return True


class MultiFeature(Feature):
    """
    Represents a vector feature in the graphical model and extracts said feature from a given data point (row
    in the dataset)
    """
    def __init__(self, name, column_names, columns, types, distribution, preprocess=None, info=None):
        self.name = name
        self.column_names = column_names
        self.columns = columns
        self.types = types
        self.distribution = distribution
        self.parameter_type = FEATURE_TYPE[distribution]
        self.preprocess = {}
        for name, pp in zip(column_names, preprocess):
            self.preprocess[name] = PREPROCESS[pp] if pp else PREPROCESS['identity']
        self.info = {}
        self.info['names'] = column_names

    def get(self, vector):
        """
        Extracts a dictionary from a row in the dataset, omitting features if they are NA in that row.
        Also converts types for each entry in the vector and applies hacky pre-processing.
        Will return an empty dictionary if all features are NA for this row.
        """
        feature = {}
        for name, column, type in zip(self.column_names, self.columns, self.types):
            if vector[column].strip() != "NA":
                feature[name] = self.preprocess[name](type(vector[column]))
        return feature

    def indicator(self, vector):
        """
        Returns True if ANY of the features in the vector are present in this particular row, False if
        all features are NA.
        """
        return any(vector[column].strip() != 'NA' for column in self.columns)


class Parameter(object):
    """
    Represents a feature parameter in the graphical model. Each (cluster, feature) tuple will have a Parameter
    assigned to it. Each parameter is a dictionary of `parameter_name`: `parameter_value` that is updated on each
    EM iteration by a `updater` instance attribute in the `update` method.

    Each parameter can also calculate the likelihood of a value as a Parameter has a particular distribution
    associated with it, seen by the subclasses of Parameter.
    """

    def __init__(self, updater, info):
        self.info = info
        self.updater = updater(info)
        self.samples = []
        self.parameters = {}

    def set_all(self, parameters):
        self.parameters = parameters

    def set(self, name, parameter):
        self.parameters[name] = parameter

    def get(self, name):
        return self.parameters[name]

    def get_all(self):
        return self.parameters

    def get_as_vector(self):
        raise NotImplementedError

    def loglikelihood(self, x):
        return self.logpdf(x)

    def likelihood(self, x):
        return np.exp(self.loglikelihood(x))

    def update(self, tau):
        """
        Called during the M step of the EM. Each updater is provided with all the
        posterior probabilities from the E step. Will silently fail
        if a parameter update fails (for numerical reasons most likely) and not
        change the parameter values for this iteration.

        tau: a column vector of all the posterior probabilities for the dataset
             calculated from the E-step
        """
        try:
            new_params = self.updater.calculate_params(tau, old=self.get_all())
            self.set_all(new_params)
            #TODO: cache things that can be calculated beforehand
        except Exception:
            # if parameter update fails, retain old parameters
            import traceback
            print traceback.format_exc()

    def handle_point(self, x, i, tau, indicator):
        """
        Called for every data point during the E-step. This method is just used as a proxy
        for the updater.
        """
        self.updater.handle_point(x, i, tau, indicator)

    def add_sample(self, x):
        """
        Used before performing MLE for this parameter. Before calling `fit()` you need to
        add samples to the parameter.
        """
        self.samples.append(x)

    def fit(self):
        """
        Will update this parameter's values to maximum likelihood estimates given the data
        in the `samples` attribute.
        """
        raise NotImplementedError

    def reset_sums(self):
        """
        Resets the data in the updater at the end of an iteration
        """
        self.updater.reset()

    def __repr__(self):
        return "Parameter(%s)" % ','.join(map(lambda x: "=".join(map(str, x)),
                                              self.parameters.items()))


class GaussianParameter(Parameter):
    """
    Parameter that has a univariate Gaussian distribution
    """
    def init_parameter(self, info):
        self.info = info
        self.set_all({
            "mu": normal(loc=0.0, scale=1),
            "sigma": 1
        })

    def plot(self, plt):
        mu, sigma = self.get('mu'), sqrt(self.get('sigma'))
        x = np.arange(mu - 3 * sigma, mu + 3 * sigma, 0.001)
        plt.plot(x, map(lambda x: np.exp(self.logpdf(x)), x))

    def logpdf(self, x):
        mu, sigma = self.get('mu'), self.get('sigma')
        temp = 2 * sigma
        return -0.5 * log(temp * pi) - (x - mu) ** 2 / temp

    def get_as_vector(self):
        mu, sigma = self.get('mu'), np.sqrt(self.get('sigma'))
        return [mu, sigma]

    def fit(self):
        self.set_all({
            "mu": np.mean(self.samples),
            "sigma": np.var(self.samples)
        })


class MultinomialParameter(Parameter):
    """
    Parameter that has a Multinomial Distribution
    """
    def init_parameter(self, info):
        self.info = info
        categories = info['categories']
        params = {}
        alpha = [1.0 / len(categories) for _ in categories]
        probabilities = dirichlet(alpha)
        for key, value in zip(categories, probabilities):
            params[key] = value
        self.set_all(params)
        self.categories = categories

    def logpdf(self, x):
        return np.log(self.get(x))

    def get_as_vector(self):
        return self.get_all().values()

    def fit(self):
        counts, params = {}, {}
        for c in self.categories:
            counts[c] = 0
        for s in self.samples:
            if s not in counts:
                counts[s] = 0
            counts[s] += 1
        for key, count in counts.items():
            params[key] = float(count) / len(self.samples)
        self.set_all(params)

    def plot(self, plt):
        labels = list(self.parameters.keys())
        parameters = [self.parameters.get(cat) for cat in labels]
        bar_width = 0.8
        index = np.arange(len(parameters))
        plt.bar(index, parameters, bar_width)
        plt.xticks(index + bar_width, labels, rotation='vertical')
        [tick.set_fontsize(5) for tick in plt.get_xticklabels()]


class MultivariateGaussianParameter(Parameter):
    """
    Parameter that has a Multivariate Gaussian Distribution
    """
    def init_parameter(self, info):
        self.info = info
        names = info['names']
        mu = {}
        sigma = {}
        for name in names:
            mu[name] = normal(loc=0.0, scale=1)
            sigma[name] = {}
            for coname in names:
                if name == coname:
                    sigma[name][coname] = 1
                else:
                    sigma[name][coname] = 0
        self.set_all({
            "mu": mu,
            "sigma": sigma
        })

    def plot(self, plt):
        pass

    def logpdf(self, x):
        """
        Takes a slice of the covariance matrix if features for this row are NA.
        Also, handles if the sliced covariance matrix is not PSD by
        setting negative eigenvalues to 0.
        """
        names, values = zip(*x.items())
        values = np.array(values)
        k = len(values)
        mu, sigma = self.get('mu'), self.get('sigma')
        mu_vec = np.array([mu[name] for name in names])
        sigma_mat = np.eye(len(names))
        for i, name in enumerate(names):
            for j, coname in enumerate(names):
                sigma_mat[i][j] = sigma[name][coname]
        sigma_mat = np.matrix(sigma_mat)
        try:
            L = np.linalg.cholesky(sigma_mat)
            log_det =  2 * np.sum(np.log(np.diag(L)))
            temp = k * log(2 * pi)
            diff = np.matrix(values - mu_vec)
            val = -0.5 * (temp  + log_det) - 0.5 * diff * np.linalg.inv(sigma_mat) * diff.T
            return val[0, 0]
        except:
            # not PSD
            w, _ = np.linalg.eig(sigma_mat)
            non_zero = w > 0
            values = values[non_zero]
            mu_vec = mu_vec[non_zero]
            sigma_mat = sigma_mat[non_zero, :][:, non_zero]
            w, _ = np.linalg.eig(sigma_mat)
            L = np.linalg.cholesky(sigma_mat)
            log_det =  2 * np.sum(np.log(np.diag(L)))
            temp = k * log(2 * pi)
            diff = np.matrix(values - mu_vec)
            val = -0.5 * (temp  + log_det) - 0.5 * diff * np.linalg.inv(sigma_mat) * diff.T
            return val[0, 0]

    def get_as_vector(self):
        mu, sigma = self.get('mu'), self.get('sigma')
        vec = []
        for name, value in mu.items():
            vec.append(value)
        for name, covar in sigma.items():
            for coname, value in sigma[name].items():
                vec.append(sigma[name][coname])
        return vec

    def fit(self):
        names = self.info['names']
        sum = {}
        sum_counts = {}
        cosum = {}
        cosum2 = {}
        cosum_counts = {}
        for name in names:
            sum[name] = 0
            sum_counts[name] = 0
            cosum[name] = {}
            cosum2[name] = {}
            cosum_counts[name] = {}
            for coname in names:
                cosum[name][coname] = 0
                cosum2[name][coname] = [0, 0]
                cosum_counts[name][coname] = 0
        for sample in self.samples:
            for name, value in sample.items():
                sum[name] += value
                sum_counts[name] += 1
                for coname, covalue in sample.items():
                    cosum[name][coname] += value * covalue
                    cosum2[name][coname][0] += value
                    cosum2[name][coname][1] += covalue
                    cosum_counts[name][coname] += 1
        mu = {}
        sigma = {}
        for name in names:
            if sum_counts[name] != 0:
                mu[name] = float(sum[name]) / sum_counts[name]
            else:
                mu[name] = 1.0
        for name in names:
            sigma[name] = {}
            for coname in names:
                sigma[name][coname] = cosum[name][coname] - cosum2[name][coname][0] * mu[coname] - cosum2[name][coname][1] * mu[name] + cosum_counts[name][coname] * mu[name] * mu[coname]
        for name in names:
            for coname in names:
                if cosum_counts[name][coname] != 0:
                    sigma[name][coname] /= cosum_counts[name][coname]
                else:
                    sigma[name][coname] = 1
        self.set_all({
            "mu": mu,
            "sigma": sigma
        })

FEATURE_TYPE = {
    "Gaussian": GaussianParameter,
    "Multinomial": MultinomialParameter,
    "MultivariateGaussian": MultivariateGaussianParameter
}

def get_parameter(type):
    return FEATURE_TYPE[type]

