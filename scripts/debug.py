#! /usr/bin/env python3

import logging
from typing import Callable, Dict
from dataclasses import dataclass
import traceback
from pathlib import Path

from scripts.logger import get_logger

logger = get_logger("debug")

@dataclass
class DebugCase:
    """Class to represent a debug test case"""
    name: str
    description: str
    function: Callable
    enabled: bool = True

class DebugRunner:
    """Class to manage and run debug test cases"""
    
    def __init__(self):
        self.cases: Dict[str, DebugCase] = {}
        self.debug_dir = Path("debug")
        
    def register_case(self, case_id: str, name: str, description: str, func: Callable, enabled: bool = True) -> None:
        """Register a new debug case"""
        self.cases[case_id] = DebugCase(name, description, func, enabled)
        
    def run_case(self, case_id: str) -> None:
        """Run a specific debug case"""
        if case_id not in self.cases:
            logger.error(f"Case {case_id} not found")
            return
            
        case = self.cases[case_id]
        if not case.enabled:
            logger.info(f"Case {case_id} ({case.name}) is disabled, skipping...")
            return
            
        logger.info(f"Running case {case_id}: {case.name}")
        logger.info(f"Description: {case.description}")
        logger.info("=" * 50)
        
        try:
            case.function()
        except Exception as e:
            logger.error(f"Error in case {case_id}: {str(e)}")
            logger.error(traceback.format_exc())
        
        logger.info("=" * 50)
        
    def run_all_enabled(self) -> None:
        """Run all enabled debug cases"""
        for case_id in self.cases:
            self.run_case(case_id)
            
    def get_case_dir(self, case_id: str) -> Path:
        """Get directory for debug case outputs"""
        case_dir = self.debug_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

# Define debug cases in separate module
from scripts.debug_cases import (
    test_bev_generation,
    test_dairy_masks,
    test_occ_generation,
    test_camera_projection,
    test_camera_params,
    test_segmentation_masks,
    test_bev_generator,
    test_camera_extrinsics,
    test_image_size,
    test_aws_occ_generation
)

def main():
    # Initialize debug runner
    runner = DebugRunner()
    
    # Register all test cases
    cases = [
        ("case_10", "AWS Occlusion", "Test AWS version of occlusion generation", test_aws_occ_generation),
        ("case_9", "Dairy Masks", "Test dairy environment mask generation", test_dairy_masks),
        ("case_8", "Occlusion Generation", "Test occlusion map generation", test_occ_generation),
        ("case_7", "Image Size", "Check image dimensions", test_image_size),
        ("case_6", "Camera Extrinsics", "Test camera extrinsics updates", test_camera_extrinsics),
        ("case_5", "BEV Generator", "Test BEV generation functionality", test_bev_generator),
        ("case_4", "Camera Projection", "Test pointcloud to camera projection", test_camera_projection),
        ("case_2", "Camera Parameters", "Test camera parameter functionality", test_camera_params),
        ("case_1", "Segmentation Masks", "Generate and visualize segmentation masks", test_segmentation_masks),
    ]
    
    for case_id, name, desc, func in cases:
        runner.register_case(case_id, name, desc, func)
    
    # Run specific case or all cases
    runner.run_case("case_9")  # Run specific case
    # runner.run_all_enabled()  # Or run all enabled cases

if __name__ == "__main__":
    main()
