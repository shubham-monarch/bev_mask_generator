import numpy as np
import pytest
from ..bev_generator import RotationUtils

def test_ypr_to_rotation_matrix():
    '''Test conversion from YPR angles to rotation matrix'''
    # Test case 1: Zero angles
    R = RotationUtils.ypr_to_rotation_matrix(0, 0, 0)
    assert np.allclose(R, np.eye(3))

    # Test case 2: 90 degree rotations
    R = RotationUtils.ypr_to_rotation_matrix(90, 0, 0)
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.allclose(R, expected, atol=1e-10)

def test_rotation_matrix_to_ypr():
    '''Test conversion from rotation matrix to YPR angles'''
    # Test case 1: Identity matrix
    yaw, pitch, roll = RotationUtils.rotation_matrix_to_ypr(np.eye(3))
    assert np.allclose([yaw, pitch, roll], [0, 0, 0], atol=1e-10)

    # Test case 2: 90 degree yaw
    R = RotationUtils.ypr_to_rotation_matrix(90, 0, 0)
    yaw, pitch, roll = RotationUtils.rotation_matrix_to_ypr(R)
    assert np.allclose([yaw, pitch, roll], [90, 0, 0], atol=1e-10)

def test_roundtrip_conversion():
    '''Test roundtrip conversion: YPR -> R -> YPR'''
    test_angles = [
        (30, 45, 60),
        (90, 0, 0),
        (0, 90, 0),
        (0, 0, 90),
        (180, 45, 30)
    ]
    
    for yaw, pitch, roll in test_angles:
        # Convert YPR to rotation matrix
        R = RotationUtils.ypr_to_rotation_matrix(yaw, pitch, roll)
        # Convert back to YPR
        yaw_back, pitch_back, roll_back = RotationUtils.rotation_matrix_to_ypr(R)
        # Check if we get back the original angles
        assert np.allclose([yaw, pitch, roll], [yaw_back, pitch_back, roll_back], atol=1) 