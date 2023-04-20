import numpy as np
import pandas as pd
import torch
import abc
import Clustering.utils.pytorch_utils as ptu

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from scipy.spatial.distance import cdist
from typing import Sequence


class TestRetestBaseEvaluator(abc.ABC):
    def __init__(self, test_labels: Sequence[np.ndarray], test_centroids: Sequence[np.ndarray], 
                 retest_labels: Sequence[np.ndarray], retest_centroids: Sequence[np.ndarray], params: dict):
        """
        test_labels, retest_labels: List[(D, H, W)]
        test_centroids, retest_centroids: List[(num_clusters, D_feature)]
        params: hausdorff_dist_pct = 95, num_clusters = 7, centroid_dist_order = 2, standard_space_atlases: List[str]
        .metric_df: colnames: dsc, hausdorff_dist, centroid_dist
        """
        self.params = params
        self.test_centroids = test_centroids
        self.retest_centroids = retest_centroids
        
        self.post_processor = AsDiscrete(to_onehot=self.params["num_clusters"] + 1)  # including bg
        self.test_label2retest_label = [np.zeros((self.params["num_clusters"] + 1,), dtype=int) for _ in range(len(test_labels))]  # to be computed; np.ndarray: e.g. 0 -> 0, 1 -> 2, ...
        self.compute_label_mapping()
        self.__align_centroids()

        self.test_labels = []
        self.retest_labels = []
        for test_label, retest_label, mapping in zip(test_labels, retest_labels, self.test_label2retest_label):
            test_label = self.align_labels(test_label, mapping)
            self.test_labels.append(self.post_processor(test_label[None, ...]))  # (D, H, W) -> (num_clusters + 1, D, H, W)
            self.retest_labels.append(self.post_processor(retest_label[None, ...]))
        self.test_labels = torch.tensor(np.stack(self.test_labels, axis=0))  # (N, num_clusters + 1, D, H, W)
        self.retest_labels = torch.tensor(np.stack(self.retest_labels, axis=0))  # (N, num_clusters + 1, D, H, W)
        self.metric_df = pd.DataFrame()
        self.dice_metric = DiceMetric(include_background=False, reduction="none")
        self.hausdorff_dist_metric = HausdorffDistanceMetric(include_background=False, percentile=self.params["hausdorff_dist_pct"], reduction="none")
    
    @abc.abstractmethod
    def compute_label_mapping(self):
        raise NotImplementedError        
    
    def __align_centroids(self):
        for i, test_centroids in enumerate(self.test_centroids):
            mapping = self.test_label2retest_label[i][1:] - 1  # converted to 0-based for all fg
            self.test_centroids[i] = test_centroids[mapping, :]

    def align_labels(self, test_label: np.ndarray, mapping: np.ndarray) -> np.ndarray:
        """
        test_label: (D, H, W)
        """
        test_label_aligned = mapping[test_label.flatten()].reshape(test_label.shape)

        return test_label_aligned

    def __call__(self):
        self.compute_dsc()
        self.compute_hausdorff_dist()
        self.compute_centroid_dist()
    
    def compute_dsc(self):
        metric_vals = self.dice_metric(self.test_labels, self.retest_labels).mean(dim=-1)  # (B, num_clusters) -> (B,)
        self.metric_df["dsc"] = ptu.to_numpy(metric_vals)

    def compute_hausdorff_dist(self):
        metric_vals = self.hausdorff_dist_metric(self.test_labels, self.retest_labels).mean(dim=-1)  # (B, num_clusters) -> (B,)
        self.metric_df["hausdorff_dist"] = ptu.to_numpy(metric_vals)

    def compute_centroid_dist(self):
        # avg on number of features
        test_centroids = np.stack(self.test_centroids, axis=0)  # (N, num_clusters, D_feats)
        retest_centroids = np.stack(self.retest_centroids, axis=0)
        dist = np.linalg.norm(test_centroids - retest_centroids, ord=self.params["centroid_dist_order"], axis=-1).mean(axis=1)  # (N,)
        self.metric_df["centroid_dist"] = dist / test_centroids.shape[-1]


class TestRetestPairwiseEvaluator(TestRetestBaseEvaluator):
    def compute_label_mapping(self):
        for i, (test_centroids, retest_centroids) in enumerate(zip(self.test_centroids, self.retest_centroids)):
            dist_mat = cdist(test_centroids, retest_centroids)
            self.test_label2retest_label[i][1:] = dist_mat.argmin(axis=1) + 1
