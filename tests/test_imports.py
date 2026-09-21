import importlib

import pytest

import geoxplain

SUBPACKAGES = ["models", "explainers", "spatial", "validation", "visualization", "io", "metrics", "workflows",
               "utils", "datasets", "cli"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(f"geoxplain.{name}")


def test_public_names_resolve():
    for name in geoxplain.__all__:
        assert hasattr(geoxplain, name), name


def test_version_is_semver():
    parts = geoxplain.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_all_lists_of_subpackages_resolve():
    for name in SUBPACKAGES:
        mod = importlib.import_module(f"geoxplain.{name}")
        for exported in getattr(mod, "__all__", []):
            assert hasattr(mod, exported), f"{name}.{exported}"


def test_missing_optional_dependency_gives_clear_error():
    from geoxplain.utils import OptionalDependencyError, require

    with pytest.raises(OptionalDependencyError) as exc:
        require("definitely_not_a_module_xyz", "testing")
    assert "definitely_not_a_module_xyz" in str(exc.value)
