"""
OmniRetail - Role & Action Classifier Module with Pick vs. Put-Back Analytics
Accurately distinguishes between:
- Role: CUSTOMER / SHOPPER vs. MART STAFF / SUPERVISOR
- Micro-Actions: WALKING, BROWSING SHELF, REACHING / PICKING, INSPECTING ITEM (HELD), PUT BACK ON SHELF, MONITORING AISLE
- Pick vs. Put-Back Verification:
  - Did they pick an item? (YES / NO with pick count)
  - Did they put it back on the shelf? (YES / NO with put-back count)
  - Net Items Taken: max(0, picks - putbacks)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .config import ZoneDefinition
from .pose_analyzer import ShopperPose
from .tracker import ShopperDetection


class PersonRole(Enum):
    CUSTOMER = "CUSTOMER"
    SUPERVISOR = "SUPERVISOR / STAFF"


class DetailedAction(Enum):
    WALKING = "WALKING IN AISLE"
    BROWSING = "BROWSING SHELF"
    REACHING = "REACHING FOR SHELF"
    PICKED_ITEM = "PICKED ITEM FROM SHELF"
    ITEM_PICKED = "PICKED ITEM FROM SHELF"
    INSPECTING_ITEM = "INSPECTING ITEM (HELD)"
    PUT_BACK = "PUT BACK ON SHELF"
    MONITORING = "MONITORING AISLE"


@dataclass
class PersonBehaviorProfile:
    track_id: int
    role: PersonRole = PersonRole.CUSTOMER
    current_action: DetailedAction = DetailedAction.WALKING
    items_picked_count: int = 0
    items_putback_count: int = 0
    items_taken_count: int = 0       # Net items taken: max(0, picked - putback)
    is_holding_item: bool = False
    holding_item_frames: int = 0
    reach_frames_streak: int = 0
    in_retract_phase: bool = False
    shelf_zone_interacted: Optional[str] = None
    role_confidence: float = 0.5
    role_locked: bool = False

    # Evidence accumulators
    customer_evidence: float = 0.0
    staff_evidence: float = 0.0

    # Gait & dwell tracking
    stationary_frames: int = 0
    shelf_facing_frames: int = 0
    walkway_frames: int = 0
    last_center_pos: Optional[Tuple[int, int]] = None
    last_reach_frame: int = -100
    last_putback_frame: int = -100


class RoleAndActionClassifier:
    """
    Multi-signal classifier separating customers from mart staff/supervisors,
    and tracking complete Pick vs. Put-Back lifecycles.
    """

    def __init__(self):
        self.profiles: Dict[int, PersonBehaviorProfile] = {}

    def classify(
        self,
        frame_idx: int,
        detections: List[ShopperDetection],
        poses: Dict[int, ShopperPose],
        zones: Dict[str, ZoneDefinition],
        frame: np.ndarray,
    ) -> Dict[int, PersonBehaviorProfile]:
        """Classify each tracked person in the current frame."""
        h, w = frame.shape[:2]

        for det in detections:
            tid = det.track_id
            pose = poses.get(tid)
            x1, y1, x2, y2 = det.bbox
            bw = x2 - x1
            bh = y2 - y1
            cx, cy = det.center_pos

            if tid not in self.profiles:
                self.profiles[tid] = PersonBehaviorProfile(
                    track_id=tid,
                    role=PersonRole.CUSTOMER,
                    current_action=DetailedAction.WALKING,
                )

            profile = self.profiles[tid]

            # ------------------------------------------------------------------
            # 1. Motion & Trajectory Analysis
            # ------------------------------------------------------------------
            if profile.last_center_pos is not None:
                dx = cx - profile.last_center_pos[0]
                dy = cy - profile.last_center_pos[1]
                speed = np.sqrt(dx * dx + dy * dy)
                if speed < 3.0:
                    profile.stationary_frames += 1
                else:
                    profile.stationary_frames = max(0, profile.stationary_frames - 1)
            profile.last_center_pos = (cx, cy)

            # ------------------------------------------------------------------
            # 2. Torso Uniform Color Variance Analysis
            # ------------------------------------------------------------------
            ty1 = int(y1 + 0.15 * bh)
            ty2 = int(y1 + 0.45 * bh)
            tx1 = int(x1 + 0.20 * bw)
            tx2 = int(x2 - 0.20 * bw)

            is_uniform_like = False
            if ty2 > ty1 and tx2 > tx1 and ty1 >= 0 and ty2 <= h and tx1 >= 0 and tx2 <= w:
                torso_roi = frame[ty1:ty2, tx1:tx2]
                if torso_roi.size > 100:
                    hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)
                    h_std = float(np.std(hsv[:, :, 0]))
                    s_std = float(np.std(hsv[:, :, 1]))
                    if h_std < 14.0 and s_std < 35.0:
                        is_uniform_like = True

            # ------------------------------------------------------------------
            # 3. Arm Reach & Physical Shelf Penetration Detection
            # ------------------------------------------------------------------
            is_reaching = False
            reaching_zone_name = None

            if pose is not None:
                if pose.reaching_zone is not None:
                    is_reaching = True
                    reaching_zone_name = pose.reaching_zone
                else:
                    # Spatial reach heuristic: check if wrist extends outward toward shelf
                    for wrist, shoulder in [(pose.left_wrist, pose.left_shoulder), (pose.right_wrist, pose.right_shoulder)]:
                        if wrist and shoulder:
                            arm_len = np.sqrt((wrist[0] - shoulder[0]) ** 2 + (wrist[1] - shoulder[1]) ** 2)
