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
                    if kconfs[idx] > 0.2 and (kpts[idx][0] > 0 or kpts[idx][1] > 0):
                        return int(kpts[idx][0]), int(kpts[idx][1])
                    return None

                nose = get_kpt(0)
                l_shoulder = get_kpt(5)
                r_shoulder = get_kpt(6)
                l_wrist = get_kpt(9)
                r_wrist = get_kpt(10)

                # Determine facing orientation
                facing = self._estimate_facing(nose, l_shoulder, r_shoulder)

                # Determine if either wrist is reaching into a shelf zone
                h, w = frame.shape[:2]
                reaching_zone = self._detect_reach(l_wrist, r_wrist, zones, w, h)

                pose_map[shopper.track_id] = ShopperPose(
                    track_id=shopper.track_id,
                    left_wrist=l_wrist,
                    right_wrist=r_wrist,
                    nose=nose,
                    left_shoulder=l_shoulder,
                    right_shoulder=r_shoulder,
                    facing=facing,
                    reaching_zone=reaching_zone,
                )

        return pose_map

    def _estimate_facing(
        self,
        nose: Optional[Tuple[int, int]],
        l_shoulder: Optional[Tuple[int, int]],
        r_shoulder: Optional[Tuple[int, int]],
    ) -> str:
        """
        Estimate whether the shopper's body/head is oriented toward the left shelf,
        right shelf, or straight down the walkway.
        """
        if nose and l_shoulder and r_shoulder:
            shoulder_mid_x = (l_shoulder[0] + r_shoulder[0]) / 2
            diff = nose[0] - shoulder_mid_x
            if diff < -12:
                return "LEFT_SHELF"
            elif diff > 12:
                return "RIGHT_SHELF"
            return "WALKWAY"

        # Fallback if only nose or one shoulder is visible
        if l_shoulder and r_shoulder:
            # If left shoulder is significantly right of right shoulder or vice versa
            shoulder_dx = r_shoulder[0] - l_shoulder[0]
            if abs(shoulder_dx) < 15:
                # Body profile side-facing
                return "LEFT_SHELF" if shoulder_dx < 0 else "RIGHT_SHELF"

        return "WALKWAY"

    def _detect_reach(
        self,
        l_wrist: Optional[Tuple[int, int]],
        r_wrist: Optional[Tuple[int, int]],
        zones: Dict[str, ZoneDefinition],
        width: int,
        height: int,
    ) -> Optional[str]:
        """Check if either wrist penetrates inside a shelf zone."""
        wrists = [w for w in (l_wrist, r_wrist) if w is not None]
        if not wrists:
            return None

        for z_name, zone in zones.items():
            if not zone.is_shelf:
                continue
            x1, y1, x2, y2 = zone.get_pixel_rect(width, height)
            for wx, wy in wrists:
                if x1 <= wx <= x2 and y1 <= wy <= y2:
                    return z_name

        return None
