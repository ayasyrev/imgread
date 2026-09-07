from importlib.metadata import version


def test_module_importable():
    import imgread

    assert hasattr(imgread, "__version__")


def test_module_version_matches_package_metadata():
    import imgread

    assert imgread.__version__ == version("imgread")
