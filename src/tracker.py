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

    def track_frame(self, frame: np.ndarray) -> List[ShopperDetection]:
        """
        Run persistent tracking on an incoming video frame.
        Filters strictly for person detections (class 0).
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            classes=[0],  # Person class
            conf=self.confidence,
            verbose=False,
        )

        detections: List[ShopperDetection] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return detections

        track_ids = boxes.id.int().cpu().tolist()
        xyxy_coords = boxes.xyxy.int().cpu().tolist()
        confs = boxes.conf.float().cpu().tolist()

        for tid, box, conf in zip(track_ids, xyxy_coords, confs):
            x1, y1, x2, y2 = box
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            detections.append(ShopperDetection(
                track_id=int(tid),
                bbox=(x1, y1, x2, y2),
                confidence=float(conf),
                foot_pos=(foot_x, foot_y),
            ))

        return detections
