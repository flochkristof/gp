from setuptools import setup, find_packages

setup(name='GP',
      version='1.0.0',
      packages=find_packages(),
      install_requires=[
          "numpy",
          "matplotlib",
          "torch",
          "gpytorch"]
      )