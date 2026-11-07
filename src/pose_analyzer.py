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

        results = self.model(frame, verbose=False, conf=self.confidence)
        if not results or len(results) == 0:
            return {}

        res = results[0]
        if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
            return {}

        pose_boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        keypoints_array = res.keypoints.xy.cpu().numpy()  # shape (N, 17, 2)
        confs = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None

        pose_map: Dict[int, ShopperPose] = {}

        # Match each tracked shopper to the closest pose detection via bounding box center
        for shopper in tracked_shoppers:
            best_idx = -1
            best_dist = float("inf")
            sx, sy = shopper.center_pos

            for i, pbox in enumerate(pose_boxes):
                px = (pbox[0] + pbox[2]) / 2
                py = (pbox[1] + pbox[3]) / 2
                dist = (sx - px) ** 2 + (sy - py) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            # Threshold distance to avoid assigning unmatched faraway poses (e.g., max 150 px)
            if best_idx != -1 and best_dist < 15000 and best_idx < len(keypoints_array):
                kpts = keypoints_array[best_idx]
                kconfs = confs[best_idx] if confs is not None else np.ones(17)

                def get_kpt(idx: int) -> Optional[Tuple[int, int]]:
