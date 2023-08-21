import copy

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.utils import distance_metric, type_metric

class PyclusteringXMeans(BaseEstimator, ClusterMixin):
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

        self.cluster_centers_ = centers
        self.labels_ = self.__clusters_to_labels(clusters)
        return self

    def predict(self, X):
        return self.xmeans_instance.predict(X)

    def __clusters_to_labels(self, clusters):
        labels = np.array([None]*len(self.data))

        for i, cluster in enumerate(clusters):
            labels[cluster] = i
        
        return labels

