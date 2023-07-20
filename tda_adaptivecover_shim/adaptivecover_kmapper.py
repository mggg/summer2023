import numpy as np

from mapper_xmean_cover.mapper_xmean_cover.AdaptiveCover import xmeans_adaptive_cover

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

        self.acinstance = xmeans_adaptive_cover(
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
        centers = [(interval[1] - interval[0])/2 for interval in self.acinstance.intervals]
        return centers

    def transform(self, X ):
        cover_intervals = self.acinstance.intervals
        intervals = [
            self.transform_single(X, interval) for interval in cover_intervals
            ]
        # Clean out any empty cubes (common in high dimensions)
        intervals = [i for i in intervals if len(i)]
        return intervals
    
    def transform_single(X, data, interval):
        interval = data[(data >= interval[0]) & (data <= interval[1])]
        return interval

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
# interval is list of intervals lists with right, left endpoints
# iterate over intervals, grab all data pts in each interval
