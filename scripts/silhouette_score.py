import argparse
import numpy as np
import nibabel as nib
import os
import glob

from sklearn.metrics import silhouette_score
from collections import defaultdict
from tqdm import tqdm
from typing import List


def run_one_thalamus(feature: np.ndarray, label: np.ndarray, mask: np.ndarray) -> float:
    feature_selected = feature[mask, :]  # (N, K)
    label_selected = label[mask]  #(N,)
    score = silhouette_score(feature_selected, label_selected)

    return score


def run_one_thalamus_all_num_clusters(feature: np.ndarray, label_filenames: List[str]):
    num_clusters_all = []
    scores = []
    for label_filename in label_filenames:
        label_iter = nib.load(label_filename).get_fdata().astype(int)
        mask = (label_iter > 0)
        num_labels = np.unique(label_iter[mask]).shape[0]
        num_clusters_all.append(num_labels)
        score_iter = run_one_thalamus(feature, label_iter, mask)
        scores.append(score_iter)
    
    scores = np.array(scores)

    return scores, num_clusters_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--spatial_weight", type=float, default=0.9)
    parser.add_argument("--SH_scaler", type=float, default=0.0185)
    args_dict = vars(parser.parse_args())

    clustering_results_dir = args_dict["results_dir"]
    spatial_weight = args_dict["spatial_weight"]
    SH_scaler = args_dict["SH_scaler"]

    if not os.path.isdir(args_dict["output_dir"]):
        os.makedirs(args_dict["output_dir"])

    # Using the original data directory
    all_subjects_dir = glob.glob(os.path.join(args_dict["data_dir"], "*"))
    avg_all_runs = defaultdict(list)
    for subject_dir_iter in tqdm(all_subjects_dir, desc="Computing silhouette score"):
        subject_id = os.path.basename(subject_dir_iter)
        for run_dirname in glob.glob(os.path.join(subject_dir_iter, "*")):
            run_dirname_basename = os.path.basename(run_dirname)
            save_dir = os.path.join(args_dict["output_dir"], subject_id, run_dirname_basename)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            feature_filename = os.path.join(run_dirname, "spherical_coeffs.nii.gz")
            label_filenames = {
                    "left": sorted(glob.glob(os.path.join(clustering_results_dir, f"*/{subject_id}/{run_dirname_basename}/left_atlas_not_remapped.nii.gz"))),
                    "right": sorted(glob.glob(os.path.join(clustering_results_dir, f"*/{subject_id}/{run_dirname_basename}/right_atlas_not_remapped.nii.gz")))
                    }
            print(label_filenames)

            feature = nib.load(feature_filename)  # (H, W, D, K)

            feature_in = feature.get_fdata()

            H, W, D = feature_in.shape[:3]
            ref = min(H, W, D)
            spatial_coords = np.meshgrid(np.linspace(0, H / ref, H), np.linspace(0, W / ref, W), np.linspace(0, D / ref, D))
            spatial_coords = np.stack(spatial_coords, axis=-1)
            feature_in = np.concatenate([np.sqrt(spatial_weight) * spatial_coords, np.sqrt(1 - spatial_weight) * SH_scaler * feature_in], axis=-1)  # (H, W, D, K + 3)
            
            out_dict = {}
            avg_score = None

            for key in label_filenames:
                scores, num_clusters = run_one_thalamus_all_num_clusters(feature_in, label_filenames[key])

                out_dict[key] = {
                        "num_clusters": num_clusters,
                        "scores": scores
                        }
                if avg_score is None:
                    avg_score = np.zeros_like(scores)
                avg_score += scores

            avg_score /= 2

            with open(os.path.join(save_dir, "silhouette_score.txt"), "w") as wf:
                wf.write(f"{out_dict}\n")
                wf.write(f"Average score: {avg_score}")

            avg_all_runs[run_dirname_basename].append(avg_score)
    
    with open(os.path.join(args_dict["output_dir"], "silhouette_score.txt"), "w") as wf:
        for key, val in avg_all_runs.items():
            wf.write(f"{key}: {sum(val) / len(val)}\n")
