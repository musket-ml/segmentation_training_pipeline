from setuptools import setup
import setuptools
setup(name='segmentation_pipeline',
      version='0.17',
      description='Declaraqtively configured pipeline for image segmentation',
      url='https://github.com/petrochenko-pavel-a/segmentation_training_pipeline',
      author='Petrochenko Pavel',
      author_email='petrochenko.pavel.a@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=["numpy", "scipy","Pillow", "cython","matplotlib", "scikit-image","tensorflow>=1.3.0","keras>=2.0.8","imageio",
"opencv-python",
"h5py",
"imgaug",
"segmentation_models"],
      zip_safe=False)