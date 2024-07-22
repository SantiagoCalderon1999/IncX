import sys

def test_python_version_check():
    # Given
    
    # When
    python_version = sys.version_info
    
    # Then
    assert python_version.major == 3
    assert python_version.minor >= 8
    assert python_version.micro >= 0
    assert python_version.releaselevel == 'final'
    assert python_version.serial >= 0