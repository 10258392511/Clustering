import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import glob

from tqdm import tqdm


def compute_dist_one_thalamus(features: np.ndarray, labels: np.ndarray) -> float:
    mask = (labels > 0)
    features = features[mask, :]
    labels = labels[mask]
    all_labels = np.unique(labels)
    error = 0

    for label_iter in all_labels:
        mask = (labels == label_iter)
        features_iter = features[mask, :]  # (N', K)
        features_centroid = features_iter.mean(axis=0, keepdims=True)  # (1, K)
        features_dist = features_iter - features_centroid  # (N', K)
        dist = (features_dist ** 2).sum()
        error += dist

    # error /= features.shape[0]

    return error, all_labels.shape[0]


def compute_dist(features: np.ndarray, labels: dict) -> int:
    """
    labels: 
    {
        left: (N',),
        right: (N'.)
    }
    """
    avg_dist = 0
    for key in labels:
        dist, num_labels = compute_dist_one_thalamus(features, labels[key])
        avg_dist += dist

    avg_dist /= 2

    return avg_dist, num_labels


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
    for subject_dir_iter in tqdm(all_subjects_dir, desc="Creating scree plot"):
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
            for i, (label_left_filename, label_right_filename) in enumerate(zip(label_filenames["left"], label_filenames["right"])):
                labels_iter = {
                        "left": nib.load(label_left_filename).get_fdata(),
                        "right": nib.load(label_right_filename).get_fdata()
                        }

                dist, num_labels = compute_dist(feature_in, labels_iter)
                out_dict[num_labels] = dist

            fig, axis = plt.subplots(figsize=(18, 7.2))
            num_labels = sorted(list(out_dict.keys()))
            dists = [out_dict[key] for key in num_labels]
            axis.plot(num_labels, dists)
            axis.set_title("Scree Plot")
            axis.set_xlabel("Number of Labels")
            axis.set_ylabel("Error")
            fig.savefig(os.path.join(save_dir, "scree_plot.png"))

            with open(os.path.join(save_dir, "scree_plot.txt"), "w") as wf:
                wf.write(f"Number of labels: {num_labels}\n")
                wf.write(f"Distances: {dists}")
