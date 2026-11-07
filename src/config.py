"""
OmniRetail - Configuration Module
Defines paths, model settings, shelf zones, and behavioral thresholds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
OUTPUT_DIR = DATA_DIR / "output_analytics"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model Weights
DETECTION_MODEL_PATH = MODELS_DIR / "yolo26n.pt"
POSE_MODEL_PATH = MODELS_DIR / "yolo26n-pose.pt"


@dataclass
class ZoneDefinition:
    """Represents a store shelf or walkway region defined as normalized coordinates (0.0 to 1.0)."""
    name: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    color: Tuple[int, int, int]  # BGR for OpenCV
    is_shelf: bool = True

    def get_pixel_rect(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized (0-1) bounds to absolute pixel coords (x1, y1, x2, y2)."""
        return (
            int(self.x_min * width),
            int(self.y_min * height),
            int(self.x_max * width),
            int(self.y_max * height),
        )


