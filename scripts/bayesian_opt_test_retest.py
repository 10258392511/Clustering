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
import glob

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
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from collections import defaultdict
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--FSLDIR", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--spatial_type", choices=["coord", "dist"])
    parser.add_argument("--max_log_SH", type=float, default=4.)
    parser.add_argument("--init", default="k-means++", choices=["k-means++", "random", "histology_atlas"])
    parser.add_argument("--n_init", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--num_SH_features", type=int, default=28)
    parser.add_argument("--temp_dir", default="../temp")
    parser.add_argument("--output_dir", default="../outputs_clustering")
    # Bayesian optimization
    parser.add_argument("--criterion", choices=["dsc_two_runs", "dsc_avg_histology"], default="dsc_two_runs")
    parser.add_argument("--init_points", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=30)
    parser.add_argument("--random_state", type=int, default=0)
    args_dict = vars(parser.parse_args())

    # Setup
    os.environ["FSLDIR"] = args_dict["FSLDIR"]
    task_name = "test_retest_kmeans"
    config_dict = load_config(task_name)
    # Overwrite the default parameters in config
    keys_to_update = ["spatial_type", "num_SH_features"]
    config_dict["features"].update({key_iter: args_dict[key_iter] for key_iter in keys_to_update})
    keys_to_update = ["init", "n_init", "max_iter"]
    config_dict["clustering"]["params"].update({key_iter: args_dict[key_iter] for key_iter in keys_to_update})
    config_dict["init"] = args_dict["init"]

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
    # end of setup

    all_subject_dirs = glob.glob(os.path.join(args_dict["data_dir"], "*"))
    
    def run_experiment_all_dsc(log_SH_coeff: float, spatial_weight: float, if_save=True):
        keys = ["dsc_two_runs", "dsc_run_A", "dsc_run_B"]
        dsc_dict = defaultdict(list)
        
        for data_dir in tqdm(all_subject_dirs, desc="subject dirs", leave=True):
            try:
                # Subject directory info
                subject_dir = os.path.basename(data_dir)
                subject_dir_abs = os.path.join(args_dict["output_dir"], subject_dir)
                if not os.path.isdir(subject_dir_abs):
                    os.makedirs(subject_dir_abs)

                # Feature engineering
                config_dict["features"]["spherical_scale"] = 10 ** log_SH_coeff
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

                for key, dsc_iter in zip(keys, [dsc_two_runs, dsc_run_A, dsc_run_B]):
                    dsc_dict[key].append(dsc_iter["whole"])

                if if_save:
                    save_dir = os.path.join(subject_dir_abs, f"SH_{10 ** log_SH_coeff: .2e}_spatial_w_{spatial_weight: .2f}".replace(".", "_"))
                    if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)

                    for key, dsc_dict_iter in zip(keys, [dsc_two_runs, dsc_run_A, dsc_run_B]):
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
           
            except Exception as e:
                print(f"data_dir: {data_dir}", file=log_file)
                print(e, file=log_file)
        
        for key, dsc_list in dsc_dict.items():
            dsc_dict[key] = sum(dsc_list) / len(dsc_list)
        
        if if_save:
            save_dict_pkl_and_txt(dsc_dict, args_dict["output_dir"], "dsc_avg")
        
        return dsc_dict

    def run_experiment(log_SH_coeff: float, spatial_weight: float):
        dsc_dict = run_experiment_all_dsc(log_SH_coeff, spatial_weight, False)
        if args_dict["criterion"] == "dsc_two_runs":
            return dsc_dict["dsc_two_runs"]
        else:
            out_val = 0.5 * (dsc_dict["dsc_run_A"] + dsc_dict["dsc_run_B"])

            return out_val
    
    pbounds = {"log_SH_coeff": (0., args_dict["max_log_SH"]), "spatial_weight": (0., 1.)}
    opt = BayesianOptimization(
        f=run_experiment,
        pbounds=pbounds,
        random_state=args_dict["random_state"]
    )
    logger = JSONLogger(path=os.path.join(args_dict["output_dir"], "logs"))
    opt.subscribe(Events.OPTIMIZATION_STEP, logger)

    opt.maximize(init_points=args_dict["init_points"], n_iter=args_dict["n_iter"])
    
    # Save results for the best hyper-parameter combination
    run_experiment_all_dsc(**opt.max["params"], if_save=True)
    
    log_file.close()