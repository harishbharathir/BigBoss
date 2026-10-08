"""Quantitative accuracy evaluation script for cross-camera Re-ID."""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path when running via python scripts/evaluate_reid.py
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 1. Define Ground Truth: list of vehicles that genuinely crossed from Gate Cam to Rear Cam
# Format: {"true_vehicle": "Silver Sedan", "gate_time": (approx_start, approx_end), "rear_time": (approx_start, approx_end)}
GROUND_TRUTH_HANDOVERS = [
    {"vehicle_name": "Silver Car", "gate_exit_range": (3.0, 7.0), "rear_entry_range": (6.0, 10.0)},
]

def evaluate_handovers(detected_handovers: list, ground_truth: list):
    tp = 0
    fp = 0
    matched_gt = set()

    for ho in detected_handovers:
        # Check if ho matches any ground truth item
        matched = False
        for idx, gt in enumerate(ground_truth):
            if (gt["gate_exit_range"][0] <= ho.exit_time <= gt["gate_exit_range"][1] and
                gt["rear_entry_range"][0] <= ho.entry_time <= gt["rear_entry_range"][1]):
                tp += 1
                matched_gt.add(idx)
                matched = True
                break
        if not matched:
            fp += 1  # False alarm (matched two unrelated cars)

    fn = len(ground_truth) - len(matched_gt)  # True handovers missed by the model

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("================ RE-ID ACCURACY REPORT ================")
    print(f"Ground Truth Handovers : {len(ground_truth)}")
    print(f"Detected Handovers     : {len(detected_handovers)}")
    print(f"True Positives (TP)    : {tp}")
    print(f"False Positives (FP)   : {fp}")
    print(f"False Negatives (FN)   : {fn}")
    print(f"-------------------------------------------------------")
    print(f"PRECISION              : {precision * 100:.2f}%")
    print(f"RECALL                 : {recall * 100:.2f}%")
    print(f"F1-SCORE               : {f1 * 100:.2f}%")
    print("=======================================================")

if __name__ == "__main__":
    from app.reid.cross_camera import analyze_multiple_cameras
    from app.retrieval.visual_search import load_visual_models

    models = load_visual_models()
    sources = [
        {"name": "Gate Cam", "source": "data/videos/gate_cam.mp4", "time_offset": 0.0},
        {"name": "Rear Cam", "source": "data/videos/rear_cam.mp4", "time_offset": 0.0},
    ]

    print("Running multi-camera analysis...")
    results = analyze_multiple_cameras(
        camera_sources=sources,
        detector=models[0],
        processor=models[1],
        embedder=models[2],
        device=models[3],
        sample_fps=2.5,
        similarity_threshold=0.70,
    )

    evaluate_handovers(results["handovers"], GROUND_TRUTH_HANDOVERS)
