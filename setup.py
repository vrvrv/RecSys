#nsml: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

from distutils.core import setup

setup(
    install_requires=[
        'pytorch-lightning',
        'numpy',
        'pandas',
        'tqdm',
        'scikit-learn'
    ]
)