import os

from Clustering.clustering import KMeansCluster
from Clustering.clustering.base_clustering import BaseCluster
from yaml import load, Loader


ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)

# mapping task/dataset_name
CONFIG_PATH = {
    "test_retest_kmeans": os.path.join(ROOT, "configs", "test_retest_kmeans.yml")
}

CLUSTERING_MAPPING = {
    "k_means": KMeansCluster
}


def load_config(task_name: str) -> dict:
    config_path = CONFIG_PATH[task_name]
    with open(config_path, "r") as rf:
        config_dict = load(rf, Loader=Loader)

    return config_dict


def load_model(task_name: str):
    """
    Returns model constructor.
    """
    config_dict = load_config(task_name)
    clustering_name = config_dict["clustering"]["name"]

    return CLUSTERING_MAPPING[clustering_name]
