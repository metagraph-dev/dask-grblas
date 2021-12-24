import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption("--runveryslow", action="store_true", help="run very slow tests")


def pytest_runtest_setup(item):
    if (
        "slow" in item.keywords
        and not item.config.getoption("--runslow")
        and not item.config.getoption("--runveryslow")
    ):
        pytest.skip("need --runslow or --runveryslow option to run")
    if "veryslow" in item.keywords and not item.config.getoption("--runveryslow"):
        pytest.skip("need --runveryslow option to run")
