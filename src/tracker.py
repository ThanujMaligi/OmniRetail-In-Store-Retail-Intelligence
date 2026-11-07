"""
OmniRetail - Shopper Tracker Module
Integrates Ultralytics YOLO26 with ByteTrack for persistent multi-shopper identification.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ultralytics import YOLO


@dataclass
class ShopperDetection:
    """Represents a single tracked shopper in a frame."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    foot_pos: Tuple[int, int]       # (x, y) bottom-center ground position

    @property
    def center_pos(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)


class ShopperTracker:
    """
    Persistent tracker wrapper around YOLO26 and ByteTrack.
    Maintains continuous track IDs across frame occlusions.
    """

    def __init__(self, model_path: str, confidence: float = 0.25, tracker: str = "bytetrack.yaml"):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.tracker = tracker
