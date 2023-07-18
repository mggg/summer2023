import numpy as np

from mapper_xmean_cover.cover import Cover
from mapper_xmean_cover.AdaptiveCover import xmeans_adaptive_cover,adaptive_cover_BFS,adaptive_cover_DFS,adaptive_cover_randomized, BIC_Cover_Centroid

class KMapperAdaptiveCover():
    def __init__(
            self,
            X,
            lens,
            initial_cover,
            clusterer,
            iterations=10, 
            max_intervals=10, 
            BIC=True, 
            delta=0., 
            method='BFS', 
            debug=None
    ):
        self.X = X
        self.lens = lens
        self.initial_cover = initial_cover
        self.clusterer = clusterer
        self.iterations = iterations
        self.max_intervals = max_intervals
        self.BIC = BIC
        self.delta = delta
        self.method = method
        self.debug = debug

        self.adaptivecoverinstance = xmeans_adaptive_cover(
            X = self.X,
            lens = self.lens,
            initial_cover = self.initial_cover,
            clusterer = self.clusterer,
            iterations = self.iterations,
            max_intervals = self.max_intervals,
            BIC = self.BIC,
            delta = self.delta,
            method = self.method,
            debug = self.debug
        )  
    def fit(self, X):
        return BIC_Cover_Centroid
    # need to call interval_centers somehow, interval_centers is generated when computing the bic presplit
    # interval_centers is a list of centers
    # it's used in bic_centroid to compute bic/aic for each iteration
    # specifically, we need the most recent computation of interval_centers
    def transform(self, X, ):
        centers = centers or self.centers_
        hypercubes = [
            self.transform_single(X, cube, i) for i, cube in enumerate(centers)
            ]
        # Clean out any empty cubes (common in high dimensions)
        hypercubes = [cube for cube in hypercubes if len(cube)]
        return hypercubes
    # this is how transform is done in kmapper.cover, i'm pretty sure this is not how to do it for the shim but my brain is broken

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# class that implements fit, transform, and fit_transform
# lie
# pass in whatever is needed to initialize adaptive cover class
# -> fit returns list of centers
# transform returns list of data points in each element of the cover
# init ac object with everything needed for ac
# self.ac = instance of ac class, then extract info for fit and transform
