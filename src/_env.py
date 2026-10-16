"""Use Pixi locally and isolated PyPI environments on Outerbounds."""

import os

from metaflow import pypi

METAFLOW_PACKAGE_POLICY = "include"


def step_env(packages, python):
    if os.environ.get("CHURN_LOCAL") == "1":
        return lambda step: step
    return pypi(packages=packages, python=python)
