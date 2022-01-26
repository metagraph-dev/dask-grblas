from setuptools import setup

import versioneer

setup(
    name="dask-grblas",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Python interface to GraphBLAS and distributed via Dask",
    author="Erik Welch",
    url="https://github.com/eriknw/dask-grblas",
    packages=["dask_grblas"],
    license="Apache License 2.0",
    setup_requires=[],
    install_requires=["grblas >= 1.3.15", "dask[array]"],
    tests_require=["pytest"],
)
