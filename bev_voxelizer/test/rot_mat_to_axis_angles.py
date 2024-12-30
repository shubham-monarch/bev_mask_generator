import numpy as np
import pytest
from bev_voxelizer.bev_generator import RotationUtils

def test_identity_matrix():
    """Test identity matrix conversion to axis angles."""
    rot_utils = RotationUtils()
    R = np.eye(3)
    angles = rot_utils.rotation_matrix_to_axis_angles(R)
    assert np.allclose(angles, (0, 0, 0), atol=1e-6)

def test_90_degree_x_rotation():
    """Test 90 degree x-axis rotation conversion."""
    rot_utils = RotationUtils()
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    angles = rot_utils.rotation_matrix_to_axis_angles(R)
    assert np.allclose(angles[0], 0, atol=1e-6)  # x-axis unchanged
    assert np.allclose(angles[1:], (90, 90), atol=1e-6)  # y,z rotated 90°

def test_90_degree_y_rotation():
    """Test 90 degree y-axis rotation conversion."""
    rot_utils = RotationUtils()
    R = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    angles = rot_utils.rotation_matrix_to_axis_angles(R)
    assert np.allclose(angles[1], 0, atol=1e-6)  # y-axis unchanged
    assert np.allclose((angles[0], angles[2]), (90, 90), atol=1e-6)  # x,z rotated 90°

def test_90_degree_z_rotation():
    """Test 90 degree z-axis rotation conversion."""
    rot_utils = RotationUtils()
    R = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    angles = rot_utils.rotation_matrix_to_axis_angles(R)
    assert np.allclose(angles[2], 0, atol=1e-6)  # z-axis unchanged
    assert np.allclose(angles[:2], (90, 90), atol=1e-6)  # x,y rotated 90°

def test_invalid_matrix():
    """Test error handling for invalid rotation matrix."""
    rot_utils = RotationUtils()
    R = np.array([[1, 0], [0, 1]])  # 2x2 matrix
    with pytest.raises(AssertionError):
        rot_utils.rotation_matrix_to_axis_angles(R)

def test_arbitrary_rotation():
    """Test 45 degree x-axis rotation conversion."""
    rot_utils = RotationUtils()
    # 45° rotation around x-axis
    angle = np.pi/4
    R = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    angles = rot_utils.rotation_matrix_to_axis_angles(R)
    assert np.allclose(angles[0], 0, atol=1e-6)  # x-axis unchanged
    assert np.allclose(angles[1:], (45, 45), atol=1e-6)  # y,z rotated 45° 