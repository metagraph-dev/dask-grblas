import os
import subprocess

import grblas.tests as gb_tests

XFAIL_TESTS = {
    "test_matrix.py": {
        "test_from_values_scalar": "Needs investigated",
        "test_resize": "Needs investigated",
        "test_build": "Needs investigated",
        "test_build_scalar": "Needs investigated",
        "test_extract_values": "Needs investigated",
        "test_extract_element": "Needs investigated",
        "test_remove_element": "Needs investigated",
        "test_extract_row": "Needs investigated",
        "test_extract_column": "Needs investigated",
        "test_extract_input_mask": "Needs investigated",
        "test_assign": "Needs investigated",
        "test_subassign_row_col": "Needs investigated",
        "test_subassign_matrix": "Needs investigated",
        "test_assign_row_scalar": "Needs investigated",
        "test_assign_row_col_matrix_mask": "Needs investigated",
        "test_assign_column_scalar": "Needs investigated",
        "test_assign_bad": "Needs investigated",
        "test_reduce_agg": "Needs investigated",
        "test_reduce_agg_argminmax": "Needs investigated",
        "test_reduce_agg_firstlast": "Needs investigated",
        "test_reduce_agg_firstlast_index": "Needs investigated",
        "test_reduce_agg_empty": "Needs investigated",
        "test_transpose": "Needs investigated",
        "test_kronecker": "Needs investigated",
        "test_assign_transpose": "Needs investigated",
        "test_transpose_equals": "Needs investigated",
        "test_transpose_exceptional": "Needs investigated",
        "test_bad_init": "Needs investigated",
        "test_equals": "Needs investigated",
        "test_bad_update": "Needs investigated",
        "test_del": "Needs investigated",
        "test_import_export": "Needs investigated",
        "test_import_on_view": "Needs investigated",
        "test_import_export_empty": "Needs investigated",
        "test_import_export_auto": "Needs investigated",
        "test_no_bool_or_eq": "Needs investigated",
        "test_bool_eq_on_scalar_expressions": "Needs investigated",
        "test_bool_eq_on_scalar_expressions_no_auto": "Needs investigated",
        "test_contains": "Needs investigated",
        "test_iter": "Needs investigated",
        "test_diag": "Needs investigated",
        "test_split": "Needs investigated",
        "test_concat": "Needs investigated",
        "test_nbytes": "Needs investigated",
        "test_auto": "Needs investigated",
        "test_auto_assign": "Needs investigated",
        "test_expr_is_like_matrix": "Needs investigated",
        "test_flatten": "Needs investigated",
        "test_autocompute_argument_messages": "Needs investigated",
        "test_infix_sugar": "Needs investigated",
        "test_random": "Needs investigated",
        "test_firstk": "Needs investigated",
        "test_lastk": "Needs investigated",
        "test_deprecated": "Needs investigated",
        "test_ndim": "Expressions need .ndim",
        "test_compactify": "Needs investigated",
        "test_sizeof": "Needs investigated",
    },
    "test_op.py": {
        "test_semiring_parameterized": "Needs investigated",
    },
    "test_resolving.py": {
        "test_bad_extract_with_updater": "Needs investigated",
        "test_updater_on_rhs": "Needs investigated",
        "test_py_indices": "Needs investigated",
    },
    "test_scalar.py": {
        "test_update": "Needs investigated",
        "test_expr_is_like_scalar": "Needs investigated",
        "test_ndim": "Needs investigated",
        "test_cscalar": "Should work in upcoming grblas release",
    },
    "test_vector.py": {
        "test_resize": "Needs investigated",
        "test_build": "Needs investigated",
        "test_build_scalar": "Needs investigated",
        "test_ewise_add": "Needs investigated",
        "test_extract_input_mask": "Needs investigated",
        "test_remove_element": "Needs investigated",
        "test_assign": "Needs investigated",
        "test_assign_scalar": "Needs investigated",
        "test_assign_scalar_mask": "Needs investigated",
        "test_assign_scalar_with_mask": "Needs investigated",
        "test_del": "Needs investigated",
        "test_import_export": "Needs investigated",
        "test_import_export_auto": "Needs investigated",
        "test_contains": "Needs investigated",
        "test_iter": "Needs investigated",
        "test_vector_index_with_scalar": "Needs investigated",
        "test_diag": "Needs investigated",
        "test_nbytes": "Needs investigated",
        "test_inner": "Needs investigated",
        "test_outer": "Needs investigated",
        "test_auto": "Needs investigated",
        "test_auto_assign": "Needs investigated",
        "test_expr_is_like_vector": "Needs investigated",
        "test_random": "Needs investigated",
        "test_firstk": "Needs investigated",
        "test_lastk": "Needs investigated",
        "test_largestk": "Needs investigated",
        "test_smallestk": "Needs investigated",
        "test_concat": "Needs investigated",
        "test_split": "Needs investigated",
        "test_ndim": "Expressions need .ndim",
        "test_compactify": "Needs investigated",
        "test_sizeof": "Needs investigated",
        "test_extract_negative_indices": "Needs investigated",
    },
    "test_numpyops.py": {"test_npunary": "sometimes works, sometimes fails?"},
}
NOT_STRICT = {
    "test_numpyops.py": {"test_npunary"},
    "test_scalar.py": {"test_cscalar"},
}


def main():
    src_folder = os.path.dirname(gb_tests.__file__)
    dst_folder = os.path.dirname(__file__)
    for filename, xfail in XFAIL_TESTS.items():
        with open(os.path.join(src_folder, filename)) as f:
            lines = f.readlines()

        def fixline(line):
            if line.startswith("def test_") and line[4:].split("(", 1)[0] in xfail:
                key = line[4:].split("(", 1)[0]
                msg = f"{xfail[key]!r}"
                is_strict = filename not in NOT_STRICT or key not in NOT_STRICT[filename]
                return f"@pytest.mark.xfail({msg!r}, strict={is_strict})\n{line}"
            if (
                line.startswith("from grblas import ")
                and ("Matrix" in line or "Vector" in line or "Scalar" in line)
                or line.startswith("from grblas.expr import Updater")
            ):
                return line[: len("from ")] + "dask_" + line[len("from ") :]
            return line

        with open(os.path.join(dst_folder, filename), "w") as f:
            f.write("".join(fixline(line) for line in lines))
    # Prettify the imports
    subprocess.check_call(["isort", dst_folder])


if __name__ == "__main__":
    main()
