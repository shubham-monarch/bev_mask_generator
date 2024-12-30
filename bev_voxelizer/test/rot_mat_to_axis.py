import numpy as np
import pytest
from bev_voxelizer.bev_generator import BEVGenerator

def test_identity_matrix():
    '''Test with identity matrix - should return 0 degrees for all axes'''
    bev_gen = BEVGenerator()
    R = np.eye(3)
    angles = bev_gen.rotation_matrix_to_axis_angles(R)
    assert np.allclose(angles, (0, 0, 0), atol=1e-6)

def test_90_degree_x_rotation():
    '''Test 90 degree rotation around x-axis'''
    bev_gen = BEVGenerator()
    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    angles = bev_gen.rotation_matrix_to_axis_angles(R)
    assert np.allclose(angles[0], 0, atol=1e-6)  # x-axis unchanged
    assert np.allclose(angles[1:], (90, 90), atol=1e-6)  # y,z rotated 90°

def test_90_degree_y_rotation():
    '''Test 90 degree rotation around y-axis'''
    bev_gen = BEVGenerator()
    R = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    angles = bev_gen.rotation_matrix_to_axis_angles(R)
    assert np.allclose(angles[1], 0, atol=1e-6)  # y-axis unchanged
    assert np.allclose((angles[0], angles[2]), (90, 90), atol=1e-6)  # x,z rotated 90°

def test_90_degree_z_rotation():
    '''Test 90 degree rotation around z-axis'''
    bev_gen = BEVGenerator()
    R = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    angles = bev_gen.rotation_matrix_to_axis_angles(R)
    assert np.allclose(angles[2], 0, atol=1e-6)  # z-axis unchanged
    assert np.allclose(angles[:2], (90, 90), atol=1e-6)  # x,y rotated 90°

def test_invalid_matrix():
    '''Test with invalid rotation matrix'''
    bev_gen = BEVGenerator()
    R = np.array([[1, 0], [0, 1]])  # 2x2 matrix
    with pytest.raises(AssertionError):
        bev_gen.rotation_matrix_to_axis_angles(R)

def test_arbitrary_rotation():
    '''Test with arbitrary rotation matrix'''
    bev_gen = BEVGenerator()
    # 45° rotation around x-axis
    angle = np.pi/4
    R = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    angles = bev_gen.rotation_matrix_to_axis_angles(R)
    assert np.allclose(angles[0], 0, atol=1e-6)  # x-axis unchanged
    assert np.allclose(angles[1:], (45, 45), atol=1e-6)  # y,z rotated 45° 