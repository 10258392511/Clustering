import sys
import os

PATH = os.path.abspath(__file__)
for _ in range(3):
    PATH = os.path.dirname(PATH)
if PATH not in sys.path:
    sys.path.append(PATH)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import re
import glob

from Clustering.configs import load_config
from Clustering.utils.utils import read_data, print_data_dict_shape
from Clustering.feature_engineering import SpatialSphericalFeature
from sklearn.cluster import KMeans
from tqdm import trange
from typing import Union


base_path = "/media/m-nas2/outputs_clustering/output_test_retest/clustering_kmeans"  # clustering results directory
input_dir = "/home/hui/hui/grid_search6"  # data directory
pattern = "pp10"
spatial_weight = 0.4
SH_coeff = 0.00021
num_SH_features = 45
max_num_clusters = 6
num_runs = 20
config_dict = load_config("test_retest_kmeans")
config_dict["features"]["spherical_scale"] = SH_coeff
config_dict["features"]["spatial_weight"] = spatial_weight
config_dict["features"]["num_SH_features"] = num_SH_features


def split_mask(features_dict: dict, labels: Union[np.ndarray, None]):
    """
    Split the labels into left and right thalamus.

    out_dict:
    {
        left: {
            features: (N, 3),
            labels: (N,)
        },
        right: ...
    }
    """
    out_dict = {}
    for key in features_dict:
        features_dict_iter = features_dict[key]  # e.g. left thalamus
        features = features_dict_iter["features"]  # (N, num_feats)
        coords = features_dict_iter["coords"]  # (N, 3)
        if labels is None:
            labels_iter = np.ones((features.shape[0],)).astype(int)
        else:
            labels_iter = labels[coords[:, 0], coords[:, 1], coords[:, 2]]
        out_dict[key] = {
                "features": features,
                "labels": labels_iter
        }
    
    return out_dict


def generate_uniform(features_dict: dict):
    """
    features_dict: see output of split_mask(.)

    output_dict:
    {
        left: {
            features: (N, 3)
        }
        right: ...
    }
    """
    out_dict = {}
    for key in features_dict:
        features = features_dict[key]["features"]  # (N, num_feats)
        features_min, features_max = features.min(axis=0, keepdims=True), features.max(axis=0, keepdims=True)  # both: (1, num_feats)
        samples = np.random.rand(*features.shape) * (features_max - features_min) + features_min
        out_dict[key] = {"features": samples}
    
    return out_dict


def cluster_features(features_dict: dict, config_dict: dict, num_clusters: int):
    """
    features_dict: see output of generate_uniform(.)
    """
    clustering_params = config_dict["clustering"]["params"]
    clustering_params["n_clusters"] = num_clusters
    for key in features_dict:
        model = KMeans(**clustering_params)
        labels = model.fit_predict(features_dict[key]["features"]) + 1  # (N, num_feat) -> (N,)
        features_dict[key]["labels"] = labels

    return features_dict


def compute_log_wss(features_dict: dict):
    """
    features_dict: see output of split_mask(.)
    """
    log_wss_avg = 0
    for key in features_dict:
        wss_iter = 0
        features = features_dict[key]["features"]  # (N, num_feats)
        labels = features_dict[key]["labels"]  # (N,)
        unique_labels = np.unique(labels)
        for label_iter in unique_labels:
            if label_iter == 0:
                continue
            mask = (labels == label_iter)
            features_selected = features[mask, :]
            wss_iter += np.sum((features_selected - features_selected.mean(axis=0, keepdims=True)) ** 2) / mask.shape[0]
        log_wss_avg += np.log(wss_iter)

    return log_wss_avg / len(features_dict.keys())


def compute_gap_stats_subject(subject_id: str, config_dict: dict, kmax: int, num_runs: int = 20):
    input_dir_iter = os.path.join(input_dir, subject_id)
    data_dict_all = read_data(input_dir_iter, if_read_tfm=False)
    features_dict = None
    for run_dir in data_dict_all:
        featurizer = SpatialSphericalFeature(config_dict)
        features_dict = featurizer(data_dict_all[run_dir])
        break
    
    # Compute wss_data
    wss_data = []
    for k in range(1, max_num_clusters + 1):
        if k == 1:
            labels = None
        else:
            label_path = f"{base_path}/kmeans-{k}/{subject_id}/run_A/atlas_not_remapped.nii.gz"
            labels = nib.load(label_path).get_fdata()
        features_standard_dict = split_mask(features_dict, labels)
        wss_iter = compute_log_wss(features_standard_dict)
        wss_data.append(wss_iter)

    # Compute wss_rand_data
    samples_df = pd.DataFrame(index=range(1, max_num_clusters + 1), columns=range(num_runs))
    for k in trange(1, max_num_clusters + 1, leave=False):
        for b in trange(num_runs, leave=False):
            samples = generate_uniform(features_dict)
            features_rand_dict = cluster_features(samples, config_dict, k)
            wss_iter = compute_log_wss(features_rand_dict)
            samples_df.loc[k, b] = wss_iter
    
    # Save results
    results_df = pd.DataFrame(index=range(1, max_num_clusters + 1))
    results_df["wss_data"] = wss_data
    results_df["wss_rand_mean"] = samples_df.mean(axis=1)
    results_df["wss_rand_std"] = samples_df.std(axis=1) * np.sqrt(1 + 1 / num_runs)
    results_df["gap"] = results_df.wss_rand_mean - results_df.wss_data
    results_df["gap_rhs"] = (results_df.gap - results_df.wss_rand_std).shift(-1)
    results_df.dropna(inplace=True)
    mask = results_df.gap >= results_df.gap_rhs
    optimal_K = -1
    if mask.sum() > 0:
        optimal_K = results_df.index[mask][0]

    return optimal_K, results_df


if __name__ == "__main__":
    subjects_path = glob.glob(f"{base_path}/*/pp10/run_A/atlas_not_remapped.nii.gz")
    all_subjects_id = set()
    for subject_file in subjects_path:
        patient_id = re.search(pattern, subject_file)[0]
        all_subjects_id.add(patient_id)
    all_subjects_id = list(all_subjects_id)

    optimal_K_all = []
    for subject_id in all_subjects_id:
        optimal_K, results = compute_gap_stats_subject(subject_id, config_dict, max_num_clusters, num_runs=num_runs)
        optimal_K_all.append(optimal_K)
        save_dir = os.path.join(os.path.dirname(base_path), "gap_stat")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, f"results.csv")
        results.to_csv(filename)
        results["gap"].plot()
        plt.savefig(os.path.join(save_dir, f"results.png"))
        with open(os.path.join(save_dir, "optimal_K.txt"), "w") as wf:
            wf.write(f"Optimal K: {optimal_K}\n")

