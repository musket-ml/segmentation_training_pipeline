from setuptools import setup
import setuptools
setup(name='segmentation_pipeline',
      version='0.21',
      description='Declaraqtively configured pipeline for image segmentation',
      url='https://github.com/petrochenko-pavel-a/segmentation_training_pipeline',
      author='Petrochenko Pavel',
      author_email='petrochenko.pavel.a@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      include_package_data=True,
      dependency_links=['https://github.com/aleju/imgaug'],
      install_requires=["numpy", "scipy","Pillow", "cython","pandas","matplotlib", "scikit-image","tensorflow>=1.3.0","keras>=2.2.4","imageio",
"opencv-python",
"h5py",
"segmentation_models",
"shapely"],
      zip_safe=False)