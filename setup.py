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
        "monai",
        "bayesian-optimization",
        "tqdm",
        "PyYAML",
        "notebook",
        "ipywidgets",
        "einops",
        "jaxlib",
        "jax"
    ]
)
