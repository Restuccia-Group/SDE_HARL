from setuptools import find_packages, setup

setup(
    name="harl",
    version="1.0.0",
    author="SDE-HARL Authors",
    description="PyTorch implementation of SDE-HARL Algorithms",
    url="https://github.com/Restuccia-Group/SDE_HARL",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "pyyaml>=5.3.1",
        "tensorboard>=2.2.1",
        "tensorboardX",
        "setproctitle",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
