"""
OmniRetail - Analytics Engine Module
Maintains customer state machines, computes dwell time per zone,
tracks shelf reach/touch conversion funnels, and accumulates footprint heatmaps.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .config import RetailConfig, ZoneDefinition
from .pose_analyzer import ShopperPose
from .tracker import ShopperDetection


class BehavioralState(Enum):
    PASSING = "PASSING"
    BROWSING = "BROWSING"
    INTERACTING = "INTERACTING"


@dataclass
class ShopperSession:
    """Historical timeline and engagement metrics for a single tracked customer."""
    track_id: int
    first_frame: int
    last_frame: int
    current_state: BehavioralState = BehavioralState.PASSING
    highest_state: BehavioralState = BehavioralState.PASSING
    zone_dwell_frames: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    interaction_count: int = 0
    last_interaction_frame: int = -100
    trajectories: List[Tuple[int, int]] = field(default_factory=list)
    primary_shelf: Optional[str] = None

    def total_dwell_seconds(self, fps: float) -> float:
        return (self.last_frame - self.first_frame + 1) / max(fps, 1.0)

    def shelf_dwell_seconds(self, shelf_name: str, fps: float) -> float:
        return self.zone_dwell_frames[shelf_name] / max(fps, 1.0)


class RetailAnalyticsEngine:
    """
    Core business logic engine. Processes frame detections & poses,
    updates shopper timelines, and calculates conversion analytics.
    """

    def __init__(self, config: RetailConfig, fps: float = 30.0, frame_size: Tuple[int, int] = (720, 404)):
        self.config = config
        self.fps = fps
        self.width, self.height = frame_size
        self.sessions: Dict[int, ShopperSession] = {}
        self.heatmap_acc = np.zeros((self.height, self.width), dtype=np.float32)

    def update(
        self,
        frame_idx: int,
        detections: List[ShopperDetection],
        poses: Dict[int, ShopperPose],
    ) -> Dict[int, BehavioralState]:
        """
        Process current frame: updates shopper states, zone dwell times, and heatmap.
        Returns a dictionary of track_id -> current BehavioralState.
        """
        active_states: Dict[int, BehavioralState] = {}

        for det in detections:
            tid = det.track_id
            fx, fy = det.foot_pos
            fx = max(0, min(self.width - 1, fx))
            fy = max(0, min(self.height - 1, fy))

            # Initialize or retrieve session
            if tid not in self.sessions:
                self.sessions[tid] = ShopperSession(
                    track_id=tid,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                )

            session = self.sessions[tid]
            session.last_frame = frame_idx
            session.trajectories.append((fx, fy))

            # Accumulate heatmap density
            self._accumulate_heatmap(fx, fy)

            # Determine which zone the shopper's feet currently occupy
            current_zone_name = self._get_zone_at(fx, fy)
            session.zone_dwell_frames[current_zone_name] += 1

            # Fetch pose information if available
            pose = poses.get(tid)
            is_reaching = (pose is not None and pose.reaching_zone is not None)

            # Check for shelf reach / pick interaction
            if is_reaching:
                if frame_idx - session.last_interaction_frame > self.config.interaction_cooldown_frames:
                    session.interaction_count += 1
                    session.last_interaction_frame = frame_idx
                    if session.primary_shelf is None:
                        session.primary_shelf = pose.reaching_zone

                session.current_state = BehavioralState.INTERACTING
                session.highest_state = BehavioralState.INTERACTING
            else:
                # Check for browsing / dwell engagement
                # If they have been near a shelf for more than dwell_browse_seconds
                max_shelf_dwell_sec = 0.0
                for z_name, zone in self.config.zones.items():
                    if zone.is_shelf:
                        dwell_sec = session.shelf_dwell_seconds(z_name, self.fps)
                        if dwell_sec > max_shelf_dwell_sec:
                            max_shelf_dwell_sec = dwell_sec
                            if session.primary_shelf is None and dwell_sec >= 1.0:
                                session.primary_shelf = z_name

                # Facing orientation boost: if shopper is oriented toward the shelf
                is_facing_shelf = (pose is not None and "SHELF" in pose.facing)

                if max_shelf_dwell_sec >= self.config.dwell_browse_seconds or (is_facing_shelf and max_shelf_dwell_sec >= 1.0):
                    session.current_state = BehavioralState.BROWSING
                    if session.highest_state != BehavioralState.INTERACTING:
                        session.highest_state = BehavioralState.BROWSING
                else:
                    session.current_state = BehavioralState.PASSING

            active_states[tid] = session.current_state

        return active_states

    def _get_zone_at(self, x: int, y: int) -> str:
        """Find the zone containing the point (x, y)."""
        for z_name, zone in self.config.zones.items():
            x1, y1, x2, y2 = zone.get_pixel_rect(self.width, self.height)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return z_name
        return "Walkway"

    def _accumulate_heatmap(self, x: int, y: int):
        """Add a Gaussian kernel around the foot position onto the heatmap accumulator."""
        sigma = self.config.heatmap_gaussian_sigma
        rad = sigma * 2
        x0, x1 = max(0, x - rad), min(self.width, x + rad + 1)
        y0, y1 = max(0, y - rad), min(self.height, y + rad + 1)

        if x1 > x0 and y1 > y0:
            grid_y, grid_x = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
            gaussian = np.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * (sigma ** 2)))
            self.heatmap_acc[y0:y1, x0:x1] += gaussian

    def get_kpis(self) -> Dict:
        """Calculate executive retail conversion and dwell metrics across all seen shoppers."""
        total_footfall = len(self.sessions)
        if total_footfall == 0:
            return {
                "total_footfall": 0,
                "engaged_browsers": 0,
                "shelf_interactors": 0,
                "conversion_rate_pct": 0.0,
                "avg_dwell_seconds": 0.0,
                "zone_stats": {},
            }

        browsers = sum(1 for s in self.sessions.values() if s.highest_state in (BehavioralState.BROWSING, BehavioralState.INTERACTING))
        interactors = sum(1 for s in self.sessions.values() if s.highest_state == BehavioralState.INTERACTING)
        total_dwell = sum(s.total_dwell_seconds(self.fps) for s in self.sessions.values())
        avg_dwell = total_dwell / total_footfall

        conversion_rate = (interactors / total_footfall * 100.0) if total_footfall > 0 else 0.0

        # Zone breakdowns
        zone_stats = {}
        for z_name, zone in self.config.zones.items():
            visitors_in_zone = sum(1 for s in self.sessions.values() if s.zone_dwell_frames[z_name] > 0)
            total_zone_sec = sum(s.shelf_dwell_seconds(z_name, self.fps) for s in self.sessions.values())
            avg_zone_sec = (total_zone_sec / visitors_in_zone) if visitors_in_zone > 0 else 0.0
            zone_interactions = sum(s.interaction_count for s in self.sessions.values() if s.primary_shelf == z_name)

            zone_stats[z_name] = {
                "name": zone.name,
                "is_shelf": zone.is_shelf,
                "unique_visitors": visitors_in_zone,
                "total_dwell_sec": round(total_zone_sec, 1),
                "avg_dwell_sec": round(avg_zone_sec, 1),
                "interactions": zone_interactions,
            }

        return {
            "total_footfall": total_footfall,
            "engaged_browsers": browsers,
            "shelf_interactors": interactors,
            "conversion_rate_pct": round(conversion_rate, 1),
            "avg_dwell_seconds": round(avg_dwell, 1),
            "zone_stats": zone_stats,
        }

    def generate_heatmap_overlay(self, base_image: np.ndarray) -> np.ndarray:
        """Normalize the accumulated heatmap and blend it onto a background video frame."""
        if np.max(self.heatmap_acc) == 0:
            return base_image.copy()

        norm = cv2.normalize(self.heatmap_acc, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        norm_uint8 = np.uint8(norm)
        colored_heatmap = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)

        # Blend with base frame
        alpha = self.config.heatmap_alpha
        blended = cv2.addWeighted(colored_heatmap, alpha, base_image, 1 - alpha, 0)
        return blended
