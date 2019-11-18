import io
import os
import sys

from setuptools import setup
import setuptools

here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(name='segmentation_pipeline',
      version='0.431',
      description='Segmentation support piepeline for Musket ML',
      long_description=long_description,
      url='https://github.com/musket-ml/segmentation_training_pipeline',
      author='Petrochenko Pavel',
      author_email='petrochenko.pavel.a@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      include_package_data=True,
      #dependency_links=['https://github.com/aleju/imgaug'],
      install_requires=["musket_core"],
      zip_safe=False)