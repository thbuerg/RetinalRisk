import setuptools

with open("README.md") as fh:
    long_description = fh.read()

setuptools.setup(
    name="retinalrisk",
    version="0.1.0",
    author="Benjamin Wild",
    author_email="b.w@fu-berlin.de",
    url="https://github.com/nebw/ehrgraphs/",
    description="Risk modelling using electronic health records and knowledge graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "train_gat = retinalrisk.scripts.train_gat:main",
        ]
    },
    install_requires=[
        "more-itertools",
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "lifelines",
        "wandb",
        "nodevectors",
        "matplotlib",
        "seaborn",
        "networkx",
        "torch-geometric",
        "pyarrow",
        "pytorch-lightning",
        "fairscale",
        "ray",
        "captum",
    ],
)
