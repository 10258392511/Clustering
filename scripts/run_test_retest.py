import sys
import os

PATH = os.path.abspath(__file__)
for _ in range(3):
    PATH = os.path.dirname(PATH)
if PATH not in sys.path:
    sys.path.append(PATH)

import argparse
import numpy as np
import pandas as pd
import glob
import pickle

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from Clustering.configs import load_config, load_model
from Clustering.feature_engineering import SpatialSphericalFeature
from Clustering.utils.utils import binarize_mask, save_image
from Clustering.evaluation import TestRetestPairwiseEvaluator


def process_one_subject(dir_name: str, config_dict: dict, args_dict: dict):
    """
    args_dict: task_name, output_dir
    """
    # extract features
    feature_params = config_dict["features"]
    path_dict_test = {
        "thalamus_mask": os.path.join(dir_name, "thalamus_warped.nii.gz"),
        "spherical_coeff": os.path.join(dir_name, "runA_wm_fod.nii.gz")
    }
    featurizer_test = SpatialSphericalFeature(path_dict_test, feature_params)
    featurizer_test.images["thalamus_mask"] = binarize_mask(featurizer_test.images["thalamus_mask"], 1)  # TODO: comment out
    feats_test, inds_test = featurizer_test()

    path_dict_retest = {
        "thalamus_mask": os.path.join(dir_name, "thalamus_warped.nii.gz"),
        "spherical_coeff": os.path.join(dir_name, "runB_wm_fod.nii.gz")
    }
    featurizer_retest = SpatialSphericalFeature(path_dict_retest, feature_params)
    featurizer_retest.images["thalamus_mask"] = binarize_mask(featurizer_retest.images["thalamus_mask"], 1)  # TODO: comment out
    feats_retest, inds_retest = featurizer_test()

    # clustering
    ctor = load_model(args_dict["task_name"])
    model_params = config_dict["clustering"]

    in_shape = featurizer_test.images["thalamus_mask"].shape
    model_test = ctor(in_shape, feats_test, inds_test, model_params)
    test_labels = model_test.fit_transform()  # (H, W, D)
    test_centroids = model_test.model.cluster_centers_

    in_shape = featurizer_retest.images["thalamus_mask"].shape
    model_retest = ctor(in_shape, feats_retest, inds_retest, model_params)
    retest_labels = model_retest.fit_transform()  # (H, W, D)
    retest_centroids = model_retest.model.cluster_centers_

    # save results
    subject_idx = os.path.basename(dir_name)  # subject{num}
    output_dir = os.path.join(args_dict["output_dir"], subject_idx)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    save_image(test_labels, os.path.join(output_dir, "test_atlas.nii.gz"))
    save_image(retest_labels, os.path.join(output_dir, "retest_atlas.nii.gz"))
    with open(os.path.join(output_dir, "test_centroids.pkl"), "wb") as wf:
        pickle.dump(test_centroids, wf)
    with open(os.path.join(output_dir, "retest_centroids.pkl"), "wb") as wf:
        pickle.dump(retest_centroids, wf)
    
    return test_labels, retest_labels, test_centroids, retest_centroids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="test_retest_kmeans")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="../outputs_clustering/test_retest")  # save logging here
    parser.add_argument("--if_tuning", action="store_true")
    parser.add_argument("--eval_mode", choices=["dsc", "hausdorff_dist", "centroid_dist"], default="dsc")

    args_dict = vars(parser.parse_args())
    config_dict = load_config(args_dict["task_name"])
    eval_params = {
        "hausdorff_dist_pct": 95,
        "num_clusters": 7,
        "centroid_dist_order": 2
    }

    def objective_func(spherical_scale: float, spatial_weight: float):
        config_dict_cp = config_dict.copy()
        if args_dict["if_tuning"]:
            config_dict_cp["features"].update({
                "spherical_scale": spherical_scale,
                "spatial_weight": spatial_weight
            })
        test_labels_all = []
        retest_labels_all = []
        test_centroids_all = []
        retest_centroids_all = []
        for dir_name in glob.glob(os.path.join(args_dict["input_dir"], "*")):
            test_labels, retest_labels, test_centroids, retest_centroids = process_one_subject(dir_name, config_dict, args_dict)
            test_labels_all.append(test_labels)
            retest_labels_all.append(retest_labels)
            test_centroids_all.append(test_centroids)
            retest_centroids_all.append(retest_centroids)
        evaluator = TestRetestPairwiseEvaluator(test_labels_all, test_centroids_all, retest_labels_all, retest_centroids_all, eval_params)
        metric_df = evaluator()
        metric_df.to_csv(os.path.join(args_dict["output_dir"], "metrics.csv"), index=False)

        return metric_df[args_dict["eval_mode"]].mean()

    if args_dict["if_tuning"]:
        # TODO
        pass
    
    else:
         objective_func(0., 0.)  # set at random since config_dict is used
