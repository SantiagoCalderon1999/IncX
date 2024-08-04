import requests
import torch


def test_pytorch_and_cuda_versions():
    # Given
    pytorch_min_version = "2.3.1"
    cuda_min_version = "1.18"

    pytorch_package = "torch"
    pytorch_url = f"https://pypi.org/pypi/{pytorch_package}/json"
    pytorch_response = requests.get(pytorch_url)

    if pytorch_response.status_code != 200:
        raise AssertionError(
            f"PyTorch package '{pytorch_package}' does not exist on PyPI."
        )

    pytorch_data = pytorch_response.json()
    pytorch_version = pytorch_data["info"]["version"]

    # When

    # Then
    assert (
        compare_versions(pytorch_version, pytorch_min_version) >= 0
    ), f"PyTorch version should be at least {pytorch_min_version}. Found version {pytorch_version}."

    installed_cuda_version = get_installed_cuda_version()

    assert (
        compare_versions(installed_cuda_version, cuda_min_version) >= 0
    ), f"Installed CUDA version should be at least {cuda_min_version}. Found version {installed_cuda_version}."


def compare_versions(version1, version2):
    """
    Compare two version strings.
    Returns:
        -1 if version1 < version2
         0 if version1 == version2
         1 if version1 > version2
    """
    v1 = list(map(int, version1.split(".")))
    v2 = list(map(int, version2.split(".")))
    return (v1 > v2) - (v1 < v2)


def get_installed_cuda_version():
    """
    Get the CUDA version installed with the current PyTorch installation.
    Returns:
        CUDA version as a string, e.g., '11.8'
    """
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        return cuda_version
    else:
        return "0.0"
