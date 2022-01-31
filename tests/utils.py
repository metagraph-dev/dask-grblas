import numpy as np

import grblas as gb

import dask_grblas as dgb


def compare(func, gb_args, dgb_args, gb_kwargs={}, dgb_kwargs={}, *, compute=None, errors=False):
    if compute is None:
        compute = not errors
    if type(gb_args) is not tuple:
        gb_args = (gb_args,)
        dgb_args = (dgb_args,)
    dgb_objects = [x for x in dgb_args if isinstance(x, dgb.base.BaseType)]
    dgb_objects.extend(x for x in dgb_kwargs.values() if isinstance(x, dgb.base.BaseType))
    for arg in dgb_objects:
        assert arg._meta.nvals == 0
    try:
        gb_result = func(*gb_args, **gb_kwargs)
    except Exception as exc:
        if not errors:
            raise
        gb_result = exc
    try:
        dgb_result = func(*dgb_args, **dgb_kwargs)
    except Exception as exc:
        if compute or not errors:
            raise
        dgb_result = exc
    if type(gb_result) is gb.Matrix:
        assert gb_result.nrows == dgb_result.nrows
        assert gb_result.ncols == dgb_result.ncols
    if compute:
        assert type(gb_result) is np.ndarray or dgb_result._meta.nvals == 0
        try:
            # compute everything twice to ensure nothing is unexpectedly mutated
            first_dgb_result = dgb_result.compute()  # noqa
            dgb_result = dgb_result.compute()
        except Exception as exc:
            if not errors:
                raise
            dgb_result = exc
    assert type(gb_result) == type(dgb_result), f"{type(gb_result)} - {type(dgb_result)}"
    if isinstance(gb_result, gb.base.BaseType):
        assert gb_result.dtype == dgb_result.dtype, f"{gb_result.dtype} - {dgb_result.dtype}"
        if not gb_result.isequal(dgb_result, check_dtype=True):
            print(gb_result)
            print("!=")
            print(dgb_result)
            assert False
    elif isinstance(gb_result, Exception):
        assert str(gb_result) == str(dgb_result)
    else:
        if type(gb_result) is np.ndarray:
            assert np.all(gb_result == dgb_result), (gb_result, dgb_result)
        else:
            assert gb_result == dgb_result, (gb_result, dgb_result)
    for arg in dgb_objects:
        assert arg._meta.nvals == 0
