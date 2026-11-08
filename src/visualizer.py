"""
OmniRetail - Visualizer Module
Renders commercial-grade, clutter-free AI overlays:
- 100% Unobstructed video background (NO opaque color blocks or heavy tints)
- Sleek Cyberpunk corner-bracket bounding boxes
- Ultra-compact, non-intrusive HUD pills for Role & Action
- Item Grab / Pick Verification: Did they take anything? (YES / NO)
- Skeletal reaching vectors from shoulder to wrist with pulse indicators
- Top Executive Glass HUD with live retail conversion KPIs
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .analytics_engine import BehavioralState, RetailAnalyticsEngine
from .config import RetailConfig, ZoneDefinition
from .pose_analyzer import ShopperPose
from .role_classifier import DetailedAction, PersonBehaviorProfile, PersonRole
from .tracker import ShopperDetection


# Modern Commercial Color Palette (BGR)
COLOR_CUSTOMER = (70, 225, 100)      # Neon Emerald Green
COLOR_STAFF = (255, 190, 40)         # Electric Cyan / Azure
COLOR_PICK_ALERT = (40, 60, 255)     # Vibrant Crimson / Red
COLOR_INSPECT = (0, 215, 255)        # Warm Amber
COLOR_WALK = (200, 200, 200)         # Soft Platinum
COLOR_GLASS_BG = (16, 18, 22)        # Deep Charcoal Glass


class RetailVisualizer:
    """Renders sleek, enterprise-grade retail analytics overlays onto video frames."""

    def __init__(self, config: RetailConfig, width: int, height: int):
        self.config = config
        self.width = width
        self.height = height

    def render_frame(
        self,
        frame: np.ndarray,
        detections: List[ShopperDetection],
        poses: Dict[int, ShopperPose],
        states: Dict[int, BehavioralState],
        engine: RetailAnalyticsEngine,
        profiles: Optional[Dict[int, PersonBehaviorProfile]] = None,
        fps_current: float = 0.0,
    ) -> np.ndarray:
        """Composite all visual elements onto the frame cleanly and unobtrusively."""
        vis = frame.copy()

        # 1. Draw subtle, elegant shelf boundaries (thin lines, NO solid color fills)
        self._draw_zones(vis)

        # 2. Draw subtle shopper movement trajectories (breadcrumbs)
        self._draw_trajectories(vis, engine)

        # 3. Draw Tracked Persons, Corner Brackets, Sleek HUD Pills, and Pose vectors
        for det in detections:
            tid = det.track_id
            x1, y1, x2, y2 = det.bbox

            # Fetch Profile & Role
            profile = profiles.get(tid) if profiles else None
            role = profile.role if profile else PersonRole.CUSTOMER
            action = profile.current_action if profile else DetailedAction.WALKING
            picks = profile.items_picked_count if profile else 0
            putbacks = profile.items_putback_count if profile else 0
            net_taken = profile.items_taken_count if profile else 0
            is_holding = profile.is_holding_item if profile else False

            accent_color = COLOR_STAFF if role == PersonRole.SUPERVISOR else COLOR_CUSTOMER
            if action == DetailedAction.PUT_BACK:
                box_color = (0, 215, 255)  # Amber / Gold for put back
            elif action in (DetailedAction.PICKED_ITEM, DetailedAction.INSPECTING_ITEM) or net_taken > 0:
                box_color = COLOR_PICK_ALERT
            elif action == DetailedAction.REACHING:
                box_color = (0, 160, 255)  # Orange for reaching
            else:
                box_color = accent_color

            # A. Draw Modern Corner Brackets (no heavy bulky box)
            self._draw_corner_brackets(vis, x1, y1, x2, y2, box_color)

            # B. Draw Ultra-Compact HUD Chip with Pick vs Put-Back details
            self._draw_compact_chip(
                vis, x1, y1, x2, y2, tid, role, action,
                picks, putbacks, net_taken, is_holding, accent_color
            )

            # C. Draw Skeletal Reaching Vectors
            pose = poses.get(tid)
            if pose:
                self._draw_pose_interaction(vis, pose, action)

        # 4. Top Executive HUD Banner (Glass bar, height 32px)
        vis = self._draw_top_hud(vis, engine, profiles, fps_current)

        return vis
