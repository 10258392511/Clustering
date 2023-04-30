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
- pp20/
    - pp20_A/
        - pp20_dwi_run01_A.nii.gz
        - thalamus_warp_0.3_thresh.nii.gz
        - wm_norm.nii.gz
    - pp20_B/
        - pp20_dwi_run01_B.nii.gz
        - thalamus_warp_0.3_thresh.nii.gz
        - wm_norm.nii.gz
- pp21/...

2. **The current thalamus mask is not binarized. Make sure foreground of thalamus is all voxels with intensity > 0, i.e. threshold is 0**
