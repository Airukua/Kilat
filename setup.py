from setuptools import setup

setup(
    name="kilat",
    version="0.1.0",
    description="Kilat: a lightweight transformer training and inference toolkit.",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
        "tqdm>=4.66",
        "pyarrow>=14.0",
        "triton>=2.0",
        "wandb>=0.16",
    ],
    packages=[
        "arc",
        "data",
        "training",
        "utils",
        "inference",
    ],
    include_package_data=True,
)
