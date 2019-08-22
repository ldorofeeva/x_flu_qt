from setuptools import setup

setup(name='x_flu_qt',
      version='0.1',
      description='X-ray fluorescence map visualizer',
      url='https://github.com/ldorofeeva/x_flu.git',
      author='Elisa',
      author_email='dorofeeva.l@gmail.com',
      license='MIT',
      packages=['x_flu_qt'],
      package_data={'x_flu_qt': ['data/pertxre_lines.csv']},
      include_package_data=True,
      install_requires=[
        'numpy>=1.16',
		'scipy>=1.3',
        'h5py>=2.9',
        'matplotlib>=3.1',
        'pyqt5>=5.13'],
      long_description=open('README.md').read(),
      scripts=['bin/x_flu']
      )
