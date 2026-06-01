import scenemap


def test_import_exposes_version() -> None:
    assert isinstance(scenemap.__version__, str)
