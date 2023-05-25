import os
import sys
from setuptools import setup, find_packages

print("Installing robosuiteVGB.")

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='robosuitevgb',
    version='1.0.0',
    packages=find_packages(),
    description='visual generalization benchmark based on robosuite ',
    author='yang',
)
