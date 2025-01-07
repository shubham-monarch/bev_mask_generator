import numpy as np
import pytest
from bev_mask_generator.bev_generator import RotationUtils

def test_identity_matrix():
    """Test that zero angles produce identity matrix."""
    R = RotationUtils.ypr_to_rotation_matrix(0, 0, 0)
    assert np.allclose(R, np.eye(3), atol=1e-10)

def test_90_degree_yaw():
    """Test 90 degree yaw rotation."""
    R = RotationUtils.ypr_to_rotation_matrix(90, 0, 0)
    expected = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    assert np.allclose(R, expected, atol=1e-10)

def test_90_degree_pitch():
    """Test 90 degree pitch rotation."""
    R = RotationUtils.ypr_to_rotation_matrix(0, 90, 0)
    expected = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    assert np.allclose(R, expected, atol=1e-10)

def test_90_degree_roll():
    """Test 90 degree roll rotation."""
    R = RotationUtils.ypr_to_rotation_matrix(0, 0, 90)
    expected = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    assert np.allclose(R, expected, atol=1e-10)

def test_combined_rotation():
    """Test combined rotation of all angles."""
    # Test angles
    yaw, pitch, roll = 45, 30, 60
    R = RotationUtils.ypr_to_rotation_matrix(yaw, pitch, roll)
    
    # Convert to radians for manual calculation
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    
    # Individual rotation matrices
    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])
    
    expected = np.dot(Rz, np.dot(Ry, Rx))
    assert np.allclose(R, expected, atol=1e-10)

def test_rotation_properties():
    """Test that resulting matrix has proper rotation matrix properties."""
    R = RotationUtils.ypr_to_rotation_matrix(30, 45, 60)
    
    # Test orthogonality: R * R^T = I
    RRt = np.dot(R, R.T)
    assert np.allclose(RRt, np.eye(3), atol=1e-10)
    
    # Test determinant = 1
    det = np.linalg.det(R)
    assert np.abs(det - 1.0) < 1e-10
    
    # Test that columns are unit vectors
    for i in range(3):
        col_norm = np.linalg.norm(R[:, i])
        assert np.abs(col_norm - 1.0) < 1e-10

def test_invalid_input():
    """Test error handling for invalid input types."""
    with pytest.raises((TypeError, ValueError)):
        RotationUtils.ypr_to_rotation_matrix("90", 0, 0)
    with pytest.raises((TypeError, ValueError)):
        RotationUtils.ypr_to_rotation_matrix(90, None, 0)
    with pytest.raises((TypeError, ValueError)):
        RotationUtils.ypr_to_rotation_matrix(90, 0, [0]) 