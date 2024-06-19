import os
import json
from setuptools import setup, find_packages

# load requirements
with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

version_file_path = os.path.join(os.path.dirname(__file__), "wod_predictor/VERSION")

with open(version_file_path) as f:
    VERSION = f.read().strip()

setup(
    name="wod_predictor",
    version=VERSION,
    description="Library for predicting Workout of the Day (WOD) scores for CrossFit athletes",
    url="https://github.com/hassannaveed1997/WOD-prediction",
    author="gymbros",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    python_requires=">=3.10",
    zip_safe=False,
    include_package_data=True,
)
