from src import __doc__ as package_doc


def test_imports_work():
    assert isinstance(package_doc, str)
    assert package_doc.startswith("Application package")
