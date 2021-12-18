# dask-grblas
[![Version](https://img.shields.io/pypi/v/dask-grblas.svg)](https://pypi.org/project/dask-grblas/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/metagraph-dev/dask-grblas/blob/main/LICENSE)
[![Build Status](https://github.com/metagraph-dev/dask-grblas/workflows/Test%20and%20Deploy/badge.svg)](https://github.com/metagraph-dev/dask-grblas/actions)
[![Coverage Status](https://coveralls.io/repos/github/metagraph-dev/dask-grblas/badge.svg?branch=main)](https://coveralls.io/github/metagraph-dev/dask-grblas?branch=main)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is a work in progress.  It is ready to be played with (lightly), but not used for any serious work.

[`grblas`](https://github.com/metagraph-dev/grblas/) provides a high-level syntax for writing [GraphBLAS](https://github.com/GraphBLAS/GraphBLAS-Pointers), which is a sparse linear algebra specification suitable for many graph algorithms.  The primary GraphBLAS implementation is [SuiteSparse:GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS/) by [Dr. Timothy Davis](http://faculty.cse.tamu.edu/davis/GraphBLAS.html).

`dask-grblas` mirrors the [`grblas`](https://github.com/metagraph-dev/grblas/) API and uses [`dask`](https://dask.org/) for out-of-core or distributed computation.  I hope this can be a pleasant way to scale GraphBLAS to very large problems.

Sparse arrays are hard.  Distributed sparse arrays are even harder--but interesting!  If you're curious, would like to help, or just feel like saying hello, please [leave a message](https://github.com/metagraph-dev/dask-grblas/issues)!
