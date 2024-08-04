import requests


def test_pypi_availability_check():
    # Given
    package_name = "requests"  # TODO: Update the package name
    url = f"https://pypi.org/pypi/{package_name}/json"

    # When
    response = requests.get(url)

    # Then
    assert (
        response.status_code == 200
    ), f"Package '{package_name}' does not exist on PyPI."
