# Copyright (c) 2022-2026, Robotera Isaac Lab Project.
# SPDX-License-Identifier: BSD-3-Clause

import os
from setuptools import find_packages, setup

# Extension metadata
PACKAGE_NAME = "robotera_ball_handling"
VERSION = "0.1.0"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description="Isaac Lab environments and tasks for Robotera humanoid robots & dexterous hands.",
    packages=find_packages("source"),
    package_dir={"": "source"},
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
    ],
)
