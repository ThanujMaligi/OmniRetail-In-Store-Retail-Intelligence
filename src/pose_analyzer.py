"""
OmniRetail - Pose & Interaction Analyzer Module
Uses YOLO26-Pose to extract skeletal landmarks, identify reaching actions into shelves,
and estimate shopper orientation/gaze towards products.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from .config import ZoneDefinition
from .tracker import ShopperDetection


@dataclass
class ShopperPose:
    """Skeletal landmarks and interaction signals for a single shopper."""
    track_id: int
    left_wrist: Optional[Tuple[int, int]]
    right_wrist: Optional[Tuple[int, int]]
    nose: Optional[Tuple[int, int]]
    left_shoulder: Optional[Tuple[int, int]]
    right_shoulder: Optional[Tuple[int, int]]
    facing: str  # "LEFT_SHELF", "RIGHT_SHELF", or "WALKWAY"
    reaching_zone: Optional[str] = None  # Name of shelf zone being reached into, if any


class ShopperPoseAnalyzer:
    """
    Analyzes body pose from YOLO26-Pose to detect physical reach,
    shelf touch/pick events, and facing orientation.
    """

    def __init__(self, model_path: str, confidence: float = 0.25):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def analyze_frame(
        self,
        frame: np.ndarray,
        tracked_shoppers: List[ShopperDetection],
        zones: Dict[str, ZoneDefinition],
    ) -> Dict[int, ShopperPose]:
        """
        Runs pose estimation and binds detected keypoints to tracked shoppers.
        Returns a mapping of track_id -> ShopperPose.
        """
        if not tracked_shoppers:
            return {}
