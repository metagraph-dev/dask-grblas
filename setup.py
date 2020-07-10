from setuptools import setup

setup(
    name='dask-grblas',
    version='0.0.0',
    description='Python interface to GraphBLAS and distributed via dask',
    author='Erik Welch',
    url='https://github.com/eriknw/dask-grblas',
    packages=['dask_grblas'],
    license='Apache License 2.0',
    setup_requires=[],
    install_requires=['grblas >= 1.3.0', 'dask[array]'],
    tests_require=['pytest'],
)
