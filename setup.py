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
        "monai",
        "bayesian-optimization",
        "tqdm",
        "PyYAML",
        "notebook",
        "einops",
        "jaxlib",
        "jax"
    ]
)
