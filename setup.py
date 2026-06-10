from setuptools import setup, find_packages

setup(
    name="kilat",
    version="1.3.0",
    description="Kilat: a lightweight transformer training and inference toolkit.",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "tqdm>=4.66",
        "pyarrow>=14.0",
        "triton>=2.0",
        "sentencepiece>=0.2.0",
        "numpy>=1.24",
        "pyyaml>=6.0",
    ],
    extras_require={
        "reporting": [
            "wandb>=0.16",
            "tensorboard>=2.14",
            "mlflow>=2.10",
            "comet_ml>=3.45",
        ],
    },
    packages=find_packages(
        where=".",
        include=[
            "arc",
            "arc.*",
            "configs",         
            "configs.*",       
            "data",
            "data.*",
            "training",
            "training.*",
            "utils",
            "utils.*",
            "pipeline",
            "pipeline.*",
            "generation",
            "generation.*",
            "distiliation",
            "distiliation.*",
        ],
    ),
    include_package_data=True,
)