# Fetal Brain Development Study by Clustering
First, please install [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki). We recommend running the scripts in a Linux system.

Second, for Python dependencies, we recommend using a `virtualenv`. Please first install `PyTorch` following the [official website](https://pytorch.org/). Then please install other 
dependencies by:
```bash
git clone https://github.com/10258392511/Clustering.git
cd Clustering
pip3 install -e .
```
If you have successfully installed all dependencies, you should be able to run at project root
```bash
python3 scripts/test_install.py
```
## Input Folder Structure
```
input_data_dir
 ┣ pp*
 ┃ ┣ run_A
 ┃ ┃ ┣ thalamus
 ┃ ┃ ┃ ┣ atlas2left_thalamus-warp.nii.gz
 ┃ ┃ ┃ ┣ atlas2right_thalamus-warp.nii.gz
 ┃ ┃ ┃ ┣ atlas2thalamus-initialize.mat
 ┃ ┃ ┃ ┣ atlas2thalamus-initialized.nii.gz
 ┃ ┃ ┃ ┣ atlas2thalamus-linear-transformed.nii.gz
 ┃ ┃ ┃ ┣ atlas2thalamus-linear.mat
 ┃ ┃ ┃ ┣ atlas2thalamus-warped.nii.gz
 ┃ ┃ ┃ ┣ left_nuclei_group_[1-7]_centroid.nii.gz
 ┃ ┃ ┃ ┣ left_nuclei_group_[1-7]_distance_feature.nii.gz
 ┃ ┃ ┃ ┣ left_nuclei_group_[1-7]_distance_from_centroid.nii.gz
 ┃ ┃ ┃ ┣ left_nuclei_group_[1-7]_label.nii.gz
 ┃ ┃ ┃ ┣ left_thalamus_atlas.nii.gz
 ┃ ┃ ┃ ┣ left_thalamus_atlasmask.nii.gz
 ┃ ┃ ┃ ┣ left_thalamus_nucleigroups_linear.nii.gz
 ┃ ┃ ┃ ┣ left_thalamus_nucleigroups_nonlinear.nii.gz
 ┃ ┃ ┃ ┣ right_nuclei_group_[1-7]_centroid.nii.gz
 ┃ ┃ ┃ ┣ right_nuclei_group_[1-7]_distance_feature.nii.gz
 ┃ ┃ ┃ ┣ right_nuclei_group_[1-7]_distance_from_centroid.nii.gz
 ┃ ┃ ┃ ┣ right_nuclei_group_[1-7]_label.nii.gz
 ┃ ┃ ┃ ┣ right_thalamus_atlas.nii.gz
 ┃ ┃ ┃ ┣ right_thalamus_atlasmask.nii.gz
 ┃ ┃ ┃ ┣ right_thalamus_nucleigroups_linear.nii.gz
 ┃ ┃ ┃ ┗ right_thalamus_nucleigroups_nonlinear.nii.gz
 ┃ ┃ ┣ B_to_A_left.mat
 ┃ ┃ ┣ B_to_A_left.nii.gz
 ┃ ┃ ┣ B_to_A_right.mat
 ┃ ┃ ┣ B_to_A_right.nii.gz
 ┃ ┃ ┣ dwi_run01_A.nii.gz
 ┃ ┃ ┣ spherical_coeffs.nii.gz
 ┃ ┃ ┣ thalamus_mask_left.nii.gz
 ┃ ┃ ┗ thalamus_mask_right.nii.gz
 ┃ ┗ run_B (Optional)
```
First, please put all subjects data (`pp1/, pp2/, ...`) under one folder (e.g. `input_data_dir`). Second, refer to the folder structure and put data for **at least one scan** (e.g. `run_A/`). When `run_A/` and `run_B/` are both available, you can run grid search and Bayesian optimization for hyper-parameter search.

## Example Commands
First, load `FSL` and activate your `virtualenv` by
```bash
module load fsl
source $YOUR_VIRTUALENV_NAME
```
We provide three programs. Make sure your working directory is `Clustering/`. To see all possible options for each argument, please run
```bash
python3 $SCRIPT_NAME -h
```

The first one is grid search for spatial weight &alpha; and spherical harmonic scaler s. We give an example command below.
```bash
python3 scripts/grid_search_test_retest.py \
--FSLDIR $FSLDIR \
--data_dir "./input_data_dir/" \
--output_dir "../outputs_clustering/grid_search/" \
--spatial_type "coord" \
--num_SH_scaler_steps 5 \
--num_spatial_weight_steps 5 \
--min_log_SH "-4" \
--max_log_SH 0 \
--init "k-means++" \
--n_init 10 \
--max_iter 300 \
--num_SH_features 28 \
--temp_dir "../temp"
```
We require an input data directory `./input_data_dir` from the folder structure above, an output directory `../outputs_clustering/grid_search/` and a temporary directory `../temp` for saving intermediate results from `flirt` command of `FSL`.

The second program is applying Bayesian Optimization (BO) for hyper-parameter search. An example command:
```bash
python3 scripts/bayesian_opt_test_retest.py \
--FSLDIR $FSLDIR \
--data_dir "./input_data_dir/" \
--output_dir "../outputs_clustering/BO/" \
--spatial_type "coord" \
--max_spatial_weight 0.9 \
--min_log_SH "-4" \
--max_log_SH 1 \
--temp_dir "../temp" \
--criterion "dsc_two_runs" \
--init_points 3 \
--n_iter 60 \
--init "k-means++" \
--n_init 10 \
--max_iter 300 \
--num_SH_features 28 \
--random_state 0
```

The third program runs clustering algorithm on each scan (i.e. `run_A/` and/or `run_B/`).  An example command:
```bash
python3 scripts/clustering.py \
--FSLDIR $FSLDIR \
--data_dir "./input_data_dir/" \
--output_dir "../outputs_clustering/clustering/" \
--clustering_type "GM" \
--n_clusters 7 \
--init "kmeans" \
--n_init 10 \
--max_iter 300 \
--covariance_type "diag" \
--spatial_type "coord" \
--spatial_weight 0.9 \
--spherical_scale 0.0185 \
--num_SH_features 28 \
--temp_dir "../temp"
```
For `clustering_type` we support K-Means (`kmeans`) and Gaussian Mixture (`GM`). Only when using `GM`, `covariance_type` is used. 

For clustering algorithm parameter initialization (`init`),  `kmeans` supports `k-means++`, `random` and `histology_atlas` which is using histology atlas for centroids initialization. `GM` supports `kmeans`, `random` and `histology_atlas`. The last option means using histology atlas for initialization of means of each Gaussian mixture component; the rest parameters are initialized with `kmeans`.

## Post-Processing Commands
We provide scree plot and silhouette score to evaluate a clustering result. Therefore, you can compare results with different number of clusters. An example commands:
```bash
python3 scripts/scree_plot.py \
--results_dir "../results_dir" \
--output_dir "../output_dir/scree_plot" \
--data_dir "./input_data_dir" \
--spatial_weight 0.9 \
--SH_scaler 0.0185
```
```bash
python3 scripts/silhouette_score.py \
--results_dir "../results_dir" \
--output_dir "../output_dir/silhouette_score" \
--data_dir "./input_data_dir" \
--spatial_weight 0.9 \
--SH_scaler 0.0185
```
Note that we keep the arguments for both program the same. You need to specify a `results_dir` which contains multiple directories. These directories are the results of the third program in the previous section (```python3 scripts/clustering.py ...```).  For example, you can set `n_clusters` to be 2, 3, ..., 10 and run ```python3 scripts/clustering.py ...``` to get a output folder for each `n_clusters`. Then you can move all these folders to `results_dir`.
