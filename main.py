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
    reporter = RetailReporter(output_dir)

    # Initialize Video Writer
    writer = None
    output_video_path = output_dir / "annotated_retail_analysis.mp4"
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    reference_frame = None
    cached_poses = {}
    profiles = {}

    start_time = time.time()
    frame_idx = 0

    print("\n[>>] Processing 15-second sequence with full role & pick classification...")
    try:
        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if reference_frame is None:
                reference_frame = frame.copy()

            # 1. Multi-Shopper Tracking (YOLO26 + ByteTrack)
            tracked_shoppers = tracker.track_frame(frame)

            # 2. Skeletal & Reaching Analysis (YOLO26-Pose)
            if frame_idx % args.pose_interval == 0 and len(tracked_shoppers) > 0:
                cached_poses = pose_analyzer.analyze_frame(frame, tracked_shoppers, config.zones)

            # 3. Role & Action Recognition (Customer vs Supervisor, Reaching vs Picking)
            profiles = role_classifier.classify(frame_idx, tracked_shoppers, cached_poses, config.zones, frame)

            # 4. Behavioral State Machine & Dwell Analytics
            active_states = engine.update(frame_idx, tracked_shoppers, cached_poses)

            # 5. Visualization & HUD Overlay
            elapsed = max(0.001, time.time() - start_time)
            current_fps = (frame_idx + 1) / elapsed
            vis_frame = visualizer.render_frame(
                frame,
                tracked_shoppers,
                cached_poses,
                active_states,
                engine,
                profiles=profiles,
                fps_current=current_fps,
            )

            # 6. Output sinks
            if writer:
                writer.write(vis_frame)

            if args.display:
                cv2.imshow("OmniRetail Live Monitor", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[!] Interrupted by user.")
                    break

            frame_idx += 1
            if frame_idx % 45 == 0 or frame_idx == max_frames:
                pct = (frame_idx / max_frames) * 100.0
                cust_cnt = sum(1 for p in profiles.values() if p.role == PersonRole.CUSTOMER)
                staff_cnt = sum(1 for p in profiles.values() if p.role == PersonRole.SUPERVISOR)
                picks = sum(p.items_picked_count for p in profiles.values())
                putbacks = sum(p.items_putback_count for p in profiles.values())
                net_took = sum(p.items_taken_count for p in profiles.values())
                print(
                    f"  [{pct:5.1f}%] Frame {frame_idx:3d}/{max_frames} ({frame_idx/fps:4.1f}s) | "
                    f"Cust: {cust_cnt} | Picks: {picks} | Put-Backs: {putbacks} | Net Took: {net_took} | "
                    f"Speed: {current_fps:.1f} FPS"
                )

    finally:
        cap.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"\n[+] 15-second analysis finished in {total_time:.1f}s.")

    # Export Structured Reports
    print("[*] Generating reports, heatmaps & logs...")
    if reference_frame is None:
        reference_frame = np.zeros((height, width, 3), dtype=np.uint8)

    report_paths = reporter.export_all(engine, reference_frame, profiles=profiles)
    for name, path in report_paths.items():
        print(f"    - {name.upper()}: {path}")

    # Print Final Audit Summary
    final_kpis = engine.get_kpis()
    print("\n" + "=" * 68)
    print("             15-SECOND STORE AUDIT SUMMARY REPORT")
    print("=" * 68)
    for tid, prof in profiles.items():
        if prof.items_taken_count > 0:
            took_str = f"YES ({prof.items_taken_count} Item)"
        elif prof.items_putback_count > 0:
            took_str = f"NO (Returned to Shelf - Put Back {prof.items_putback_count})"
        else:
            took_str = "NO"
        print(f"  * Person #{tid:<2d} | Role: {prof.role.value:<18} | Action: {prof.current_action.value:<22} | Took Item? {took_str}")

    total_picks = sum(p.items_picked_count for p in profiles.values())
    total_putbacks = sum(p.items_putback_count for p in profiles.values())
    total_net = sum(p.items_taken_count for p in profiles.values())
    print("-" * 68)
    print(f"  Total Unique Individuals:           {len(profiles)}")
    print(f"  Total Shelf Picks:                  {total_picks}")
    print(f"  Total Put-Backs (Returned):         {total_putbacks}")
    print(f"  Net Items Taken Away:               {total_net}")
    conv_report = "0.0% (Inspected & Returned)" if (total_putbacks > 0 and total_net == 0) else f"{final_kpis['conversion_rate_pct']}%"
    print(f"  Aisle Shopper Conversion:           {conv_report}")
    print("=" * 68)
    print(f"[*] Processed 15-second annotated video saved to:\n    {output_video_path}")
    print("[*] Open video with: start data/output_analytics/annotated_retail_analysis.mp4\n")


if __name__ == "__main__":
    main()
