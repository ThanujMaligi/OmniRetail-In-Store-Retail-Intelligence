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

    def _draw_corner_brackets(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Tuple[int, int, int],
        thickness: int = 2,
    ):
        """Draw sleek, high-tech corner brackets around the person."""
        bw = x2 - x1
        bh = y2 - y1
        corner_len = max(8, min(18, bw // 4, bh // 4))

        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)

        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)

        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)

        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)

        # Faint dashed/thin border linking corners for context
        cv2.rectangle(frame, (x1, y1), (x2, y2), (color[0] // 3, color[1] // 3, color[2] // 3), 1)

    def _draw_compact_chip(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        tid: int,
        role: PersonRole,
        action: DetailedAction,
        picks: int,
        putbacks: int,
        net_taken: int,
        is_holding: bool,
        accent_color: Tuple[int, int, int],
    ):
        """Draw an ultra-compact, non-intrusive 2-tier HUD pill showing Pick vs. Put-Back."""
        role_title = "CUSTOMER" if role == PersonRole.CUSTOMER else "MART STAFF"

        # Action tag text & color
        if action == DetailedAction.PUT_BACK:
            act_str = "PUT BACK ON SHELF"
            act_color = (60, 220, 255)       # Amber Gold
            took_str = "TOOK: NO (Returned)"
        elif action == DetailedAction.INSPECTING_ITEM:
            act_str = "INSPECTING PRODUCT"
            act_color = (0, 215, 255)        # Light Amber
            took_str = "HELD IN HAND"
        elif action == DetailedAction.PICKED_ITEM:
            act_str = "PICKED FROM SHELF"
            act_color = (40, 60, 255)        # Bright Alert
            took_str = f"Picks: {picks}"
        elif action == DetailedAction.REACHING:
            act_str = "REACHING FOR SHELF"
            act_color = (0, 160, 255)        # Orange
            took_str = "TOOK: NO"
        elif action == DetailedAction.BROWSING:
            act_str = "BROWSING SHELF"
            act_color = (0, 220, 255)
            took_str = "TOOK: NO (Returned)" if putbacks > 0 else "TOOK: NO"
        elif action == DetailedAction.MONITORING:
            act_str = "MONITORING AISLE"
            act_color = (255, 200, 50)
            took_str = "TOOK: NO"
        else:
            act_str = "WALKING"
            act_color = (200, 200, 200)
            took_str = "TOOK: NO (Returned)" if putbacks > 0 else "TOOK: NO"

        # Override took_str if net item is confirmed taken away
        if net_taken > 0 and not is_holding and action != DetailedAction.PUT_BACK:
            took_str = f"TOOK: YES ({net_taken})"

        line1 = f"#{tid} [{role_title}]"
        line2 = f"{act_str} | {took_str}"

        # Dynamic size
        chip_w = 208
        chip_h = 32

        # Position above head, or below feet if near top edge
        chip_x = max(4, min(self.width - chip_w - 4, x1))
        if y1 - chip_h - 6 > 36:
            chip_y = y1 - chip_h - 6
        else:
            chip_y = min(self.height - chip_h - 4, y2 + 6)

        # Semi-transparent glass pill background
        roi = frame[chip_y : chip_y + chip_h, chip_x : chip_x + chip_w]
        if roi.shape[0] == chip_h and roi.shape[1] == chip_w:
            pill = roi.copy()
            cv2.rectangle(pill, (0, 0), (chip_w, chip_h), COLOR_GLASS_BG, -1)
            cv2.addWeighted(pill, 0.85, roi, 0.15, 0, roi)

            # Left accent color stripe
            cv2.rectangle(frame, (chip_x, chip_y), (chip_x + 3, chip_y + chip_h), accent_color, -1)
            # Thin border outline
            cv2.rectangle(frame, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), (50, 50, 60), 1)

            # Text render
            cv2.putText(frame, line1, (chip_x + 8, chip_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, accent_color, 1, cv2.LINE_AA)
            cv2.putText(frame, line2, (chip_x + 8, chip_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.30, act_color, 1, cv2.LINE_AA)

    def _draw_zones(self, frame: np.ndarray):
        """Draw subtle, non-intrusive shelf zone boundaries (NO opaque color fills!)."""
        for _, zone in self.config.zones.items():
            if zone.is_shelf:
                x1, y1, x2, y2 = zone.get_pixel_rect(self.width, self.height)
                # Thin 1px outline
                cv2.rectangle(frame, (x1, y1), (x2, y2), zone.color, 1)

                # Shelf marker chip at the bottom
                badge_w = 180
                badge_h = 20
                badge_x = min(self.width - badge_w - 6, max(x1 + 6, 6))
                badge_y = max(y1 + 6, self.height - badge_h - 8)

                cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (20, 20, 24), -1)
                cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), zone.color, 1)
                cv2.putText(
                    frame,
                    f"SHELF: {zone.name[:20]}",
                    (badge_x + 6, badge_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    zone.color,
                    1,
                    cv2.LINE_AA,
                )

    def _draw_trajectories(self, frame: np.ndarray, engine: RetailAnalyticsEngine):
        """Draw breadcrumb trails for each active shopper."""
        for session in engine.sessions.values():
            pts = session.trajectories[-20:]
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    alpha = int(80 + 150 * (i / len(pts)))
                    color = (alpha, alpha, alpha)
                    cv2.line(frame, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

    def _draw_pose_interaction(
        self,
        frame: np.ndarray,
        pose: ShopperPose,
        action: DetailedAction,
