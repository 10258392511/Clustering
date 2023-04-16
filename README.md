# Fetal Brain Development Study by Clustering

We recommend using a `virtualenv`. Please first install `PyTorch` following the [official website](https://pytorch.org/). Then please install other 
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
1. Test-Retest Naming Convention
- pp01_dwi_run01_A.nii.gz
- pp01_dwi_run01_B.nii.gz
- runA_wm_fod.nii.gz
- runB_wm_fod.nii.gz
- thalamus_warped.nii.gz (assuming thalamus voxel intensity > 1)
