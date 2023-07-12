import sys
import os

PATH = os.path.abspath(__file__)
for _ in range(3):
    PATH = os.path.dirname(PATH)
if PATH not in sys.path:
    sys.path.append(PATH)

os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pickle

from Clustering.feature_engineering import SpatialSphericalFeature, DistSphericalFeature
from Clustering.configs import load_config
from Clustering.utils.utils import (
    read_data, 
    print_data_dict_shape, 
    save_image,
    save_dict_pkl_and_txt,
    save_prob_maps
)
from Clustering.clustering import KMeansCluster, GaussianMixtureCluster
from Clustering.evaluation.test_retest import (
    compute_dsc_two_scans,
    compute_dsc_cluster_and_histology
)
from collections import defaultdict
from functools import reduce
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--FSLDIR", required=True)
    parser.add_argument("--clustering_type", default="kmeans", choices=["kmenas", "GM"])
    parser.add_argument("--n_components", type=int, default=7)
    parser.add_argument("--init", default="k-means++", choices=["kmeans++", "random", "histology_atlas", "kmeans"])
    parser.add_argument("--n_init", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--covariance_type", default="diag", choices=["spherical", "tied", "diag", "full"])
    parser.add_argument("--spatial_weight", type=float, default=.5)
    parser.add_argument("--spherical_scale", type=float, default=55)
    parser.add_argument("--spatial_type", default="coord", choices=["coord", "dist"])
    parser.add_argument("--num_SH_features", type=int, default=28)
    parser.add_argument("--temp_dir", default="../temp")
    parser.add_argument("--output_dir", default="../outputs_clustering")
    args_dict = vars(parser.parse_args())

    # Setup
    os.environ["FSLDIR"] = args_dict["FSLDIR"]
    if args_dict["clustering_type"] == "kmeans":
        task_name = "test_retest_kmeans"
    else:
        task_name = "test_retest_GM"

    config_dict = load_config(task_name)
    # Overwrite the default parameters in config
    keys_to_update = ["spatial_type", "spherical_scale", "spatial_weight", "num_SH_features"]
    config_dict["features"].update({key_iter: args_dict[key_iter] for key_iter in keys_to_update})
    if args_dict["clustering_type"] == "kmeans":
        assert args_dict["init"] != "kmeans", f"""Only supports {["k-means++", "random", "histology_atlas"]}"""
        keys_to_update = ["n_clusters", "init", "n_init"]
    else:
        assert args_dict["init"] not in  ["k-means++", "histology_atlas"], f"""Only supports {["kmeans", "random"]}"""
        args_dict["init_params"] = args_dict["init"]
        args_dict["n_components"] = args_dict["n_clusters"]
        keys_to_update = ["n_clusters", "init_params", "n_init", "covariance_type"]

    config_dict["clustering"].update({key_iter: args_dict[key_iter] for key_iter in keys_to_update})
       
    if not os.path.isdir(args_dict["temp_dir"]):
        os.makedirs(args_dict["temp_dir"])
    if not os.path.isdir(args_dict["output_dir"]):
        os.makedirs(args_dict["output_dir"])
    log_file = open(os.path.join(args_dict["output_dir"], "exp_logging.txt"), "w")

    save_dict_pkl_and_txt(args_dict, args_dict["output_dir"], "args_dict")
    save_dict_pkl_and_txt(config_dict, args_dict["output_dir"], "config_dict")
    
    featurizer_ctor = None
    if args_dict["spatial_type"] == "coord":
        featurizer_ctor = SpatialSphericalFeature
    elif args_dict["spatial_type"] == "dist":
        featurizer_ctor = DistSphericalFeature
    log_SH_scaler_grid = np.linspace(0., args_dict["max_log_SH"], args_dict["num_SH_scaler_steps"])
    spatial_weights_grid = np.linspace(0., 1., args_dict["num_spatial_weight_steps"])
    
    clustering_ctor = None
    if args_dict["clustering_type"] == "kmeans":
        clustering_ctor = KMeansCluster
    elif args_dict["clustering_type"] == "GM":
        clustering_ctor = GaussianMixtureCluster
    # end of Setup

    all_subject_dirs = glob.glob(os.path.join(args_dict["data_dir"], "*"))
    for data_dir in tqdm(all_subject_dirs, desc="subject dirs", leave=True):
        SH_coeff = args_dict["spherical_scale"]
        spatial_weight = args_dict["spatial_weight"]
        try:
            # Subject directory info
            subject_dir = os.path.dirname(data_dir).basename(data_dir)
            subject_dir_abs = os.path.join(args_dict["output_dir"], subject_dir)
            if not os.path.isdir(subject_dir_abs):
                os.makedirs(subject_dir_abs)

            # Feature engineering
            config_dict["features"]["spherical_scale"] = SH_coeff
            config_dict["features"]["spatial_weight"] = spatial_weight

            data_dict_all = read_data(data_dir)
            for run_dir in data_dict_all:
                data_dict = data_dict_all[run_dir]
                
                featurizer = featurizer_ctor(config_dict)
                feature_dict = featurizer(data_dict)

                # Clustering
                cluster_model = clustering_ctor(config_dict)
                cluster_dict = cluster_model.fit_transform(data_dict, feature_dict)

                # Compute DSC
                num_classes = config_dict["clustering"]["params"]["n_clusters"]
                percentile = config_dict["alignment"]["hausdorff_percent"]
                dsc_run = compute_dsc_cluster_and_histology(data_dict, cluster_dict)

                # Save results
                save_dir = os.path.join(subject_dir_abs, run_dir)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                save_dict_pkl_and_txt(dsc_run, save_dir, "dsc_compared_with_histology")
                
                system_affine_mat = data_dict_all[run_dir]["left"]["thalamus_mask"].affine
                save_image(cluster_dict["left"]["atlas_not_remapped"] + cluster_dict["right"]["atlas_not_remapped"], os.path.join(save_dir, "atlas_not_remapped.nii.gz"), system_affine_mat)
                save_image(cluster_dict["left"]["atlas"] + cluster_dict["right"]["atlas"], os.path.join(save_dir, "atlas_remapped_to_histology.nii.gz"), system_affine_mat)

                save_prob_maps(save_dir, cluster_dict["left"]["prob_maps"], cluster_dict["left"]["prob_maps"], system_affine_mat)
                
        except Exception as e:
            print(f"{data_dir}", file=log_file)
            print(e, file=log_file)
