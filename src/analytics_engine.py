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
