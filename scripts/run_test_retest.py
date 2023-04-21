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
import warnings

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
    if not args_dict["if_tuning"]:
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
    parser.add_argument("--spherical_scale_log_min", type=float, default=0)
    parser.add_argument("--spherical_scale_log_max", type=float, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--if_print_in_objective_func", action="store_true")
    parser.add_argument("--init_points", type=int, default=5)
    parser.add_argument("--num_iters", type=int, default=30)

    args_dict = vars(parser.parse_args())
    config_dict = load_config(args_dict["task_name"])
    eval_params = {
        "hausdorff_dist_pct": 95,
        "num_clusters": 14,
        "centroid_dist_order": 2
    }
    if not os.path.isdir(args_dict["output_dir"]):
        os.makedirs(args_dict["output_dir"])

    with open(os.path.join(args_dict["output_dir"], "desc.txt"), "w") as wf:
        for key, val in args_dict.items():
            wf.write(f"{key}: {val}\n")
        wf.write("-" * 100)
        wf.write("\n")
        for key, val in config_dict.items():
            wf.write(f"{key}: {val}\n")

    warnings.filterwarnings("ignore")
    def objective_func(spherical_scale_log: float, spatial_weight: float):
        config_dict_cp = config_dict.copy()
        if args_dict["if_tuning"]:
            spherical_scale = 10 ** spherical_scale_log
            config_dict_cp["features"].update({
                "spherical_scale": spherical_scale,
                "spatial_weight": spatial_weight
            })
        test_labels_all = []
        retest_labels_all = []
        test_centroids_all = []
        retest_centroids_all = []
        all_dirs = glob.glob(os.path.join(args_dict["input_dir"], "*"))
        for i, dir_name in enumerate(all_dirs):
            if args_dict["if_print_in_objective_func"]:
                print(f"Current: subject {i + 1}/{len(all_dirs)}")
            test_labels, retest_labels, test_centroids, retest_centroids = process_one_subject(dir_name, config_dict, args_dict)
            test_labels_all.append(test_labels)
            retest_labels_all.append(retest_labels)
            test_centroids_all.append(test_centroids)
            retest_centroids_all.append(retest_centroids)
        evaluator = TestRetestPairwiseEvaluator(test_labels_all, test_centroids_all, retest_labels_all, retest_centroids_all, eval_params)
        metric_df = evaluator()
        metric_df.to_csv(os.path.join(args_dict["output_dir"], "metrics.csv"), index=False)

        return metric_df[args_dict["eval_mode"]].mean()

    best_spherical_scale_log = np.log10(config_dict["features"]["spherical_scale"])
    best_spatial_weight = config_dict["features"]["spatial_weight"]
    if args_dict["if_tuning"]:
        pbounds = {
            "spherical_scale_log": (args_dict["spherical_scale_log_min"], args_dict["spherical_scale_log_max"]),
            "spatial_weight": (0, 1)
        }
        opt = BayesianOptimization(objective_func, pbounds, random_state=args_dict["seed"], verbose=2)
        logger = JSONLogger(path=os.path.join(args_dict["output_dir"], "logs.json"))
        opt.subscribe(Events.OPTIMIZATION_STEP, logger)
        
        print("Performing hyperparam search...")
        opt.maximize(init_points=args_dict["init_points"], n_iter=args_dict["num_iters"])
        best_spherical_scale_log = opt.max["params"]["spherical_scale_log"]
        best_spatial_weight = opt.max["params"]["spatial_weight"]
        args_dict["if_tuning"] = False  # required, since args_dict is used in objective_func(.)
    
    with open(os.path.join(args_dict["output_dir"], "desc.txt"), "a") as wf:
        wf.write("-" * 100)
        wf.write("\n")
        wf.write(f"best_spherical_scale: {10 ** best_spherical_scale_log}, best_spatial_weight: {best_spatial_weight}\n")
    args_dict["if_print_in_objective_func"] = True
    objective_func(best_spherical_scale_log, best_spatial_weight)
