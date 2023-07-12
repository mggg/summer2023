import warnings
from collections import defaultdict
from numbers import Integral, Real
import copy

import numpy as np

from sklearn._config import config_context
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state, gen_batches
# from sklearn.utils._param_validation import Interval, validate_params
# from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.utils import distance_metric, type_metric

class PyclusteringXMeans:
    def __init__(
        self,
        *,
        initial_centers = None,
        kmax=20,
        tolerance=0.001,
        criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION,
        ccore=True,
        repeat=1,
        random_state=None,
        metric= copy.copy(distance_metric(type_metric.EUCLIDEAN_SQUARE)),
        alpha=0.9,
        beta=0.9
    ):
        self.initial_centers = initial_centers
        self.kmax = kmax
        self.tolerance = tolerance
        self.criterion = criterion
        self.ccore = ccore
        self.repeat = repeat
        self.random_state = random_state
        self.metric = metric
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y=None):
        self.data = X

        self.xmeans_instance = xmeans(
            data = X,
            initial_centers = self.initial_centers,
            kmax = self.kmax,
            tolerance = self.tolerance,
            criterion = self.criterion,
            ccore = self.ccore,
            repeat = self.repeat,
            random_state = self.random_state,
            metric = self.metric,
            alpha = self.alpha,
            beta = self.beta
        )

        self.xmeans_instance.process()
        clusters = self.xmeans_instance.get_clusters()
        centers  = self.xmeans_instance.get_centers()

        self.labels_ = self.__clusters_to_labels(clusters)

    def __clusters_to_labels(self, clusters):
        labels = np.array([None]*len(self.data))

        for i, cluster in enumerate(clusters):
            labels[cluster] = i
        
        return labels

