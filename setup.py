from setuptools import setup, find_packages

setup(
    name="Clustering",
    version="0.1",
    author="Zhexin Wu",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn",
        "nibabel",
        "monai",
        "tqdm",
        "PyYAML",
        "notebook",
        "einops",
    ]
)
