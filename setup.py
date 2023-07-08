from setuptools import setup, find_packages

setup(
    name="Clustering",
    version="0.1",
    author="Zhexin Wu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "nibabel",
        "SimpleITK",
        "monai==0.8.1",
        "bayesian-optimization==1.4.0",
        "tqdm",
        "PyYAML",
        "notebook",
        "ipywidgets",
        "einops",
        "jaxlib",
        "jax",
        "fslpy"
    ]
)
