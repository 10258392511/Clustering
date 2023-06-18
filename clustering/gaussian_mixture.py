from .base_clustering import *


class GaussianMixtureCluster(BaseCluster):
    def __init__(self, config: dict):
        super().__init__(config)
    
    def create_probabilistic_maps(self, feature_dict: dict, key: str, hausdorff_dist_df: pd.DataFrame):
        model = self.models[key]
        thalamus_feature_dict = feature_dict[key]
        coords = thalamus_feature_dict["coords"]  # (N_all,)
        atlas2standard = hausdorff_dist_df.idxmin(axis=1)
        shape = thalamus_feature_dict["shape"]
        num_clusters = model.weights_.shape[0]
        prob_maps = np.zeros((*shape, num_clusters + 1))
        feature_probs = model.predict_proba(thalamus_feature_dict["features"])  # (N_all, num_clusters)

        for label in range(num_clusters):
            tgt_channel = atlas2standard[label + 1]
            prob_maps[coords[:, 0], coords[:, 1], coords[:, 2], tgt_channel] += feature_probs[:, label]
        
        assert np.all(prob_maps <= 1 + 1e-4)
        return prob_maps
    