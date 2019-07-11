import os
from setuptools import setup

REQUIRES = [
    "torch",
    "tqdm",
    "pytorch-pretrained-bert"
]


root = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(root, "bertsum", "about.py")) as f:
    package_info = {}
    info = f.read()
    exec(info, package_info)

setup(
    name=package_info["__title__"],
    version=package_info["__version__"],
    url=package_info["__uri__"],
    description=package_info["__description__"],
    packages=[
        "bertsum", "bertsum.others", "bertsum.models", "bertsum.prepro"
        ],
    install_requires=REQUIRES,
    dependency_links=['https://github.com/pytorch/pytorch'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ]
)
