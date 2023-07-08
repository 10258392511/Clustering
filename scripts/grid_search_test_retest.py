import sys
import os

PATH = os.path.abspath(__file__)
for _ in range(3):
    PATH = os.path.dirname(PATH)
if PATH not in sys.path:
    sys.path.append(PATH)

os.environ["FSLDIR"] = "/cluster/apps/fsl/5.0.7/x86_64"
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
    save_dict_pkl_and_txt
)
from Clustering.clustering import KMeansCluster
from Clustering.evaluation.test_retest import (
    compute_dsc_two_scans,
    compute_dsc_cluster_and_histology
)
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--spatial_type", choices=["coord", "dist"])
    parser.add_argument("--num_SH_scaler_steps", type=int, default=10)
    parser.add_argument("--num_spatial_weight_steps", type=int, default=10)
    parser.add_argument("--max_log_SH", type=float, default=4.)
    parser.add_argument("--temp_dir", default="../temp")
    parser.add_argument("--output_dir", default="../outputs_clustering")
    args_dict = vars(parser.parse_args())

    # setup
    task_name = "test_retest_kmeans"
    config_dict = load_config(task_name)
    config_dict["features"]["spatial_type"] = args_dict["spatial_type"]

    if not os.path.isdir(args_dict["temp_dir"]):
        os.makedirs(args_dict["temp_dir"])
    if not os.path.isdir(args_dict["output_dir"]):
        os.makedirs(args_dict["output_dir"])

    save_dict_pkl_and_txt(args_dict, args_dict["output_dir"], "args_dict")
    save_dict_pkl_and_txt(config_dict, args_dict["output_dir"], "config_dict")
    
    featurizer_ctor = None
    if args_dict["spatial_type"] == "coord":
        featurizer_ctor = SpatialSphericalFeature
    elif args_dict["spatial_type"] == "dist":
        featurizer_ctor = DistSphericalFeature
    log_SH_scaler_grid = np.linspace(0., args_dict["max_log_SH"], args_dict["num_SH_scaler_steps"])
    spatial_weights_grid = np.linspace(0., 1., args_dict["num_spatial_weight_steps"])
    # end of setup
    
    all_subject_dirs = glob.glob(os.path.join(args_dict["data_dir"], "*"))
    for data_dir in tqdm(all_subject_dirs, desc="subject dirs", leave=True):
        df_dict = {}
        df_dict_keys = ["dsc_two_runs", "dsc_run_A", "dsc_run_B"]
        for key in df_dict_keys:
            df_dict[key] = pd.DataFrame(columns=log_SH_scaler_grid, index=spatial_weights_grid)

        for SH_coeff in tqdm(log_SH_scaler_grid, desc="SH", leave=False):
            for spatial_weight in tqdm(spatial_weights_grid, desc="spatial weights", leave=False):
                # Subject directory info
                subject_dir = os.path.basename(data_dir)
                subject_dir_abs = os.path.join(args_dict["output_dir"], subject_dir)
                if not os.path.isdir(subject_dir_abs):
                    os.makedirs(subject_dir_abs)

                # Feature engineering
                config_dict["features"]["spherical_scale"] = 10 ** SH_coeff
                config_dict["features"]["spatial_weight"] = spatial_weight

                data_dict_all = read_data(data_dir)
                data_dict_A = data_dict_all["run_A"]
                data_dict_B = data_dict_all["run_B"]
                
                featurizer = featurizer_ctor(config_dict)
                feature_dict_A = featurizer(data_dict_A)
                featurizer = featurizer_ctor(config_dict)
                feature_dict_B = featurizer(data_dict_B)

                # Clustering
                cluster_model = KMeansCluster(config_dict)
                cluster_dict_A = cluster_model.fit_transform(data_dict_A, feature_dict_A)
                cluster_model = KMeansCluster(config_dict)
                cluster_dict_B = cluster_model.fit_transform(data_dict_B, feature_dict_B)

                # Compute DSC
                num_classes = config_dict["clustering"]["params"]["n_clusters"]
                percentile = config_dict["alignment"]["hausdorff_percent"]
                dsc_two_runs = compute_dsc_two_scans(data_dict_all, cluster_dict_A, cluster_dict_B, args_dict["temp_dir"], num_classes, percentile)
                dsc_run_A = compute_dsc_cluster_and_histology(data_dict_A, cluster_dict_A)
                dsc_run_B = compute_dsc_cluster_and_histology(data_dict_B, cluster_dict_B)

                # Save results
                save_dir = os.path.join(subject_dir_abs, f"SH_{10 ** SH_coeff: .2e}_spatial_w_{spatial_weight: .2f}".replace(".", "_"))
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                for key, dsc_dict_iter in zip(df_dict_keys, [dsc_two_runs, dsc_run_A, dsc_run_B]):
                    df_dict[key].loc[spatial_weight, SH_coeff] = dsc_dict_iter["whole"]
                    save_dict_pkl_and_txt(dsc_dict_iter, save_dir, key)
                
                for key in ["run_A", "run_B"]:
                    if key == "run_A":
                        cluster_dict_iter = cluster_dict_A
                    else:
                        cluster_dict_iter = cluster_dict_B
                    system_affine_mat = data_dict_all[key]["left"]["thalamus_mask"].affine
                    
                    save_image(cluster_dict_iter["left"]["atlas_not_remapped"] + cluster_dict_iter["right"]["atlas_not_remapped"], os.path.join(save_dir, f"{key}_atlas_not_remapped.nii.gz"), system_affine_mat)
                    save_image(cluster_dict_iter["left"]["atlas"] + cluster_dict_iter["right"]["atlas"], os.path.join(save_dir, f"{key}_atlas_remapped_to_histology.nii.gz"), system_affine_mat)
                
                save_image(data_dict_all["run_A"]["B2A"]["left_B2A"] + data_dict_all["run_A"]["B2A"]["right_B2A"], os.path.join(save_dir, "B2A_direct.nii.gz"), data_dict_all["run_A"]["left"]["thalamus_mask"].affine)
        
        # Save df's and heatmaps
        for key, df_iter in df_dict.items():
            df_iter.to_csv(os.path.join(subject_dir_abs, f"{key}.csv"))
            
            fig, axis = plt.subplots()
            handle = axis.contourf(10 ** df_iter.columns, df_iter.index, df_iter.values, cmap="plasma")
            axis.set_xscale("log")
            axis.set_xlabel("SH scaler")
            axis.set_ylabel("spatial weight")
            axis.set_title("DSC")
            plt.colorbar(handle, ax=axis)
            fig.savefig(os.path.join(subject_dir_abs, f"{key}.png"))
