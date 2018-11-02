from setuptools import setup
import setuptools
setup(name='segmentation_pipeline',
      version='0.1',
      description='The funniest joke in the world',
      url='http://github.com/storborg/funniest',
      author='Petrochenko Pavel',
      author_email='petrochenko.pavel.a@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      install_requires=["numpy", "scipy","Pillow", "cython","matplotlib", "scikit-image","tensorflow>=1.3.0","keras>=2.0.8","imageio"
"opencv-python",
"h5py",
"imgaug",
"segmentation_models"],
      zip_safe=False)