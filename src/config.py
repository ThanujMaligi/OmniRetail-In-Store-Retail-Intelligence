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


@dataclass
class RetailConfig:
    """Master Configuration for the OmniRetail Engine."""
    # Detection & Tracking Parameters
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    tracker_type: str = "bytetrack.yaml"
    target_class_id: int = 0  # Person in COCO

    # Behavioral State Thresholds (in seconds)
    dwell_browse_seconds: float = 2.0      # Dwell time to qualify as "Browsing / Engaged"
    interaction_cooldown_frames: int = 15  # Cooldown between counting successive touches
    reach_distance_threshold_px: int = 25  # Max distance from shelf boundary to trigger reach

    # Heatmap Settings
    heatmap_gaussian_sigma: int = 16
    heatmap_alpha: float = 0.55

    # Store Layout (Normalized 0.0 - 1.0)
    # Default layout assumes an overhead or angled aisle camera:
    # Left 35% = Left Shelf, Center 30% = Walkway, Right 35% = Right Shelf
    zones: Dict[str, ZoneDefinition] = field(default_factory=lambda: {
        "Shelf_Left": ZoneDefinition(
            name="Snacks & Bakery (Left)",
            x_min=0.0,
            y_min=0.0,
            x_max=0.35,
            y_max=1.0,
            color=(245, 130, 48),  # Coral / Orange
            is_shelf=True,
        ),
        "Walkway": ZoneDefinition(
            name="Central Walkway",
            x_min=0.35,
            y_min=0.0,
            x_max=0.65,
            y_max=1.0,
            color=(60, 180, 75),  # Green
            is_shelf=False,
        ),
        "Shelf_Right": ZoneDefinition(
            name="Beverages & Dairy (Right)",
            x_min=0.65,
            y_min=0.0,
            x_max=1.0,
            y_max=1.0,
            color=(0, 130, 200),  # Blue
            is_shelf=True,
        ),
    })

    def configure_for_video(self, video_path: str):
        """Automatically optimize planogram zones based on camera angle & content."""
        v_name = Path(video_path).name.lower()
        if any(k in v_name for k in ("clean_grocery_picking", "shelf_pick", "clean_shopper")):
            self.zones = {
                "Walkway": ZoneDefinition(
                    name="Shopper Walkway",
                    x_min=0.0,
                    y_min=0.0,
                    x_max=0.38,
                    y_max=1.0,
                    color=(60, 180, 75),
                    is_shelf=False,
                ),
                "Grocery_Shelf": ZoneDefinition(
                    name="Packaged Goods & Beverages",
                    x_min=0.38,
                    y_min=0.0,
                    x_max=1.0,
                    y_max=1.0,
                    color=(0, 140, 255),
                    is_shelf=True,
                ),
            }
        elif "staff_customer_cctv" in v_name or "aura" in v_name:
            self.zones = {
                "Shelf_Left": ZoneDefinition(
                    name="Retail Shelf (Left)",
                    x_min=0.0,
                    y_min=0.0,
                    x_max=0.32,
                    y_max=1.0,
                    color=(245, 130, 48),
                    is_shelf=True,
                ),
                "Walkway": ZoneDefinition(
                    name="Customer Walkway",
                    x_min=0.32,
                    y_min=0.0,
                    x_max=0.68,
                    y_max=1.0,
                    color=(60, 180, 75),
                    is_shelf=False,
                ),
                "Shelf_Right": ZoneDefinition(
                    name="Aisle Display (Right)",
                    x_min=0.68,
                    y_min=0.0,
                    x_max=1.0,
                    y_max=1.0,
                    color=(0, 130, 200),
                    is_shelf=True,
                ),
            }

