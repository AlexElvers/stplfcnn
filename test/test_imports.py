import pytest

from stplfcnn import datareaders, estimators, hyperopt, partitioners, visualization


@pytest.mark.parametrize("package", [datareaders, estimators, hyperopt, partitioners, visualization],
                         ids=lambda pkg: pkg.__name__.split(".")[-1])
def test_package_classes(package):
    for class_name in package._class_modules:
        assert package.get_class(class_name).__name__ == class_name
