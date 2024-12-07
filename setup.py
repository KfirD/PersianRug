from setuptools import setup, find_packages

setup(
    name="PersianRug",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "dill==0.3.8",
        "ipython==8.12.3",
        "jaxtyping==0.2.36",
        "matplotlib==3.8.4",
        "numpy==2.1.3",
        "pandas==2.2.3",
        "scipy==1.14.1",
        "torch==2.3.0",
        "tqdm==4.66.4"
    ],
)
