import sys
import os

PATH = os.path.abspath(__file__)
for _ in range(3):
    PATH = os.path.dirname(PATH)
if PATH not in sys.path:
    sys.path.append(PATH)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import monai
import nibabel as nib
import bayes_opt
import tqdm
import yaml
import Clustering.utils.pytorch_utils as ptu

from sklearn.cluster import KMeans


if __name__ == "__main__":
    print(f"Available device: {ptu.DEVICE}")
    print("Successfully installed all dependencies!")