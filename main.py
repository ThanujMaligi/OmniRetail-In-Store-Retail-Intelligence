"""
OmniRetail - Main Execution Pipeline
Combines YOLO26 detection, ByteTrack tracking, YOLO26-Pose landmark reaching analysis,
Role Classification (Customer vs Supervisor/Staff), and Item Pick/Take detection.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    DETECTION_MODEL_PATH,
    OUTPUT_DIR,
    POSE_MODEL_PATH,
    RAW_VIDEOS_DIR,
    RetailConfig,
)
from src.tracker import ShopperTracker
from src.pose_analyzer import ShopperPoseAnalyzer
from src.analytics_engine import RetailAnalyticsEngine
from src.role_classifier import RoleAndActionClassifier, PersonRole, DetailedAction
from src.visualizer import RetailVisualizer
from src.reporter import RetailReporter


def parse_args():
    parser = argparse.ArgumentParser(description="OmniRetail: Shopper Attention, Role & Interaction AI")
    parser.add_argument(
        "--video",
        type=str,
        default=str(RAW_VIDEOS_DIR / "clean_grocery_picking.mp4"),
        help="Path to input CCTV video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help="Directory to save output analytics and annotated video",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,  # 0 = auto-calculate exact 15 seconds (15 * fps)
        help="Maximum frames to process (0 = automatically process 15 seconds based on stream FPS)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Save annotated video with retail HUD overlay",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Show live OpenCV rendering window during inference",
    )
    parser.add_argument(
        "--pose-interval",
        type=int,
        default=2,
        help="Run pose analysis every N frames to accelerate inference on CPU",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    print("=" * 68)
    print("  OMNIRETAIL - 15-SECOND RETAIL ROLE & ACTION RECOGNITION AI")
    print("  Detects: Who are they? What are they doing? Did they take an item?")
    print("  Powered by Ultralytics YOLO26 & YOLO26-Pose")
    print("=" * 68)
    print(f"[*] Input Video:     {video_path}")
    print(f"[*] Detection Model: {DETECTION_MODEL_PATH}")
    print(f"[*] Pose Model:      {POSE_MODEL_PATH}")
    print(f"[*] Output Dir:      {output_dir}")

    # Open Video Source
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Target 15 seconds: if max_frames is 0, auto-set to 15 * fps
    target_15s_frames = int(15.0 * fps)
    if args.max_frames > 0:
        max_frames = min(args.max_frames, total_frames)
    else:
        max_frames = min(target_15s_frames, total_frames)
    clip_duration_s = max_frames / fps

    print(f"[*] Stream Specs:     {width}x{height} @ {fps:.1f} FPS | Target: {max_frames} frames ({clip_duration_s:.1f}s)")

    # Initialize Modules
    config = RetailConfig()
    config.configure_for_video(str(video_path))
    tracker = ShopperTracker(str(DETECTION_MODEL_PATH), confidence=config.confidence_threshold)
    pose_analyzer = ShopperPoseAnalyzer(str(POSE_MODEL_PATH), confidence=0.25)
    role_classifier = RoleAndActionClassifier()
    engine = RetailAnalyticsEngine(config, fps=fps, frame_size=(width, height))
    visualizer = RetailVisualizer(config, width=width, height=height)
