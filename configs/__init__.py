import os

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
# from Clustering.clustering import KMeansCluster
# from Clustering.clustering.base_clustering import BaseCluster
from yaml import load, Loader


ROOT = os.path.abspath(__file__)
for _ in range(2):
    ROOT = os.path.dirname(ROOT)

# mapping task/dataset_name
CONFIG_PATH = {
    "test_retest_kmeans": os.path.join(ROOT, "configs", "test_retest_kmeans.yml"),
    "test_retest_GM": os.path.join(ROOT, "configs", "test_retest_GM.yml")
}

CLUSTERING_MODEL_MAPPING = {
    "KMeans": KMeans,
    "GM": GaussianMixture
}

# CLUSTERING_MAPPING = {
#     "k_means": KMeansCluster
# }


def load_config(task_name: str) -> dict:
    config_path = CONFIG_PATH[task_name]
    with open(config_path, "r") as rf:
        config_dict = load(rf, Loader=Loader)

    return config_dict


def load_clustering_model(config_dict: dict):
    clustering_dict = config_dict["clustering"]
    ctor = CLUSTERING_MODEL_MAPPING[clustering_dict["name"]]
    model = ctor(**clustering_dict["params"])

    return model


# # TODO: change to use config_dict as input
# def load_model(task_name: str):
#     """
#     Returns model constructor.
#     """
#     config_dict = load_config(task_name)
#     clustering_name = config_dict["clustering"]["name"]

#     return CLUSTERING_MAPPING[clustering_name]
