import numpy as np
import open3d as o3d
import pytest
from scripts.bev_generator import BEVGenerator

def create_test_pcd(points, labels):
    """Helper function to create an Open3D tensor point cloud."""
    pcd = o3d.t.geometry.PointCloud()
    pcd.point["positions"] = o3d.core.Tensor(points)
    pcd.point["label"] = o3d.core.Tensor(labels)
    return pcd

def test_points_inside_bounds():
    """Test when points generate indices within bounds."""
    generator = BEVGenerator()
    
    points = np.array([
        [1.0, 0.0, 1.0],  # should map to middle indices
    ], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    
    pcd = create_test_pcd(points, labels)
    bb = {'x_min': 0.0, 'x_max': 2.0, 'z_min': 0.0, 'z_max': 2.0}
    
    # Should not raise any assertion errors
    generator.bev_to_seg_mask_mono(pcd, nx=4, nz=4, bb=bb)

def test_points_outside_x_bounds():
    """Test when points generate x indices outside bounds."""
    generator = BEVGenerator()
    
    points = np.array([
        [3.0, 0.0, 1.0],  # x > x_max
    ], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    
    pcd = create_test_pcd(points, labels)
    bb = {'x_min': 0.0, 'x_max': 2.0, 'z_min': 0.0, 'z_max': 2.0}
    
    # Should raise AssertionError: "x-indices are out of bounds!"
    with pytest.raises(AssertionError, match="x-indices are out of bounds!"):
        generator.bev_to_seg_mask_mono(pcd, nx=4, nz=4, bb=bb)

def test_points_outside_z_bounds():
    """Test when points generate z indices outside bounds."""
    generator = BEVGenerator()
    
    points = np.array([
        [1.0, 0.0, 3.0],  # z > z_max
    ], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    
    pcd = create_test_pcd(points, labels)
    bb = {'x_min': 0.0, 'x_max': 2.0, 'z_min': 0.0, 'z_max': 2.0}
    
    # Should raise AssertionError: "z-indices are out of bounds!"
    with pytest.raises(AssertionError, match="z-indices are out of bounds!"):
        generator.bev_to_seg_mask_mono(pcd, nx=4, nz=4, bb=bb)

def test_points_negative_coords():
    """Test when points have negative coordinates."""
    generator = BEVGenerator()
    
    points = np.array([
        [-1.0, 0.0, -1.0],  # both x,z < min
    ], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    
    pcd = create_test_pcd(points, labels)
    bb = {'x_min': 0.0, 'x_max': 2.0, 'z_min': 0.0, 'z_max': 2.0}
    
    # Should raise AssertionError for x-indices first
    with pytest.raises(AssertionError, match="x-indices are out of bounds!"):
        generator.bev_to_seg_mask_mono(pcd, nx=4, nz=4, bb=bb)

def test_points_exactly_at_bounds():
    """Test when points are exactly at bounding box limits."""
    generator = BEVGenerator()
    
    points = np.array([
        [2.0, 0.0, 2.0],  # exactly at max bounds
    ], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    
    pcd = create_test_pcd(points, labels)
    bb = {'x_min': 0.0, 'x_max': 2.0, 'z_min': 0.0, 'z_max': 2.0}
    
    # Should raise AssertionError since index will be equal to nx/nz
    with pytest.raises(AssertionError):
        generator.bev_to_seg_mask_mono(pcd, nx=4, nz=4, bb=bb) 