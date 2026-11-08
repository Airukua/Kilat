from setuptools import setup, find_packages

setup(
    name="kilat",
    version="1.0.0",
    description="Kilat: kernelized lightweight attention.",
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
        "dev": [
            "ruff>=0.4.0",
            "mypy>=1.10.0",
            "pytest>=8.0.0",
            "pre-commit>=3.5.0",
            "types-PyYAML>=6.0",
            "types-tqdm>=4.66",
        ],
    },
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={
        "console_scripts": [
            "kilat=kilat.cli:main",
        ],
    },
    include_package_data=True,
)
