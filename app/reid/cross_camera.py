"""Multi-camera vehicle tracking, visual Re-ID, and cross-camera handover detection.

Allows ingesting multiple CCTV camera feeds or recordings, detecting vehicles,
extracting deep visual embeddings with SigLIP, and tracking when a vehicle exits
one camera and reappears on another camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from ultralytics import YOLO

from app.retrieval.visual_search import load_visual_models


def extract_dominant_color(crop: np.ndarray) -> str:
    """Classify the dominant color of a vehicle crop using HSV color spaces."""
    if crop is None or crop.size == 0:
        return "unknown"
    try:
        # Crop inner 70% to avoid background/road bleed
        h, w = crop.shape[:2]
        if h > 10 and w > 10:
            dy, dx = int(h * 0.15), int(w * 0.15)
            core = crop[dy : h - dy, dx : w - dx]
        else:
            core = crop

        hsv = cv2.cvtColor(core, cv2.COLOR_RGB2HSV)
        v_mean = float(np.mean(hsv[:, :, 2]))
        s_mean = float(np.mean(hsv[:, :, 1]))
        h_mean = float(np.mean(hsv[:, :, 0]))

        if v_mean < 55:
            return "black"
        if v_mean > 190 and s_mean < 40:
            return "white"
        if s_mean < 50:
            return "silver/gray"
        if h_mean < 15 or h_mean >= 165:
            return "red"
        if 15 <= h_mean < 35:
            return "yellow/orange"
        if 35 <= h_mean < 85:
            return "green"
        if 85 <= h_mean < 135:
            return "blue"
        return "other"
    except Exception:
        return "unknown"


@dataclass
class SingleCameraTrack:
    """A vehicle tracklet detected and tracked within a single camera stream."""

    track_id: int
    camera_name: str
    label: str
    dominant_color: str
    start_time: float
    end_time: float
    duration: float
    best_conf: float
    best_time: float
    best_crop: np.ndarray
    obs_count: int
    is_moving: bool = False
    left_camera: bool = False
    entered_camera: bool = False
    embedding: np.ndarray | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    @property
    def time_range_str(self) -> str:
        return f"{self.start_time:.1f}s – {self.end_time:.1f}s"


@dataclass
class CrossCameraHandover:
    """Represents a vehicle leaving one camera and appearing on another."""

    global_id: int
    from_camera: str
    to_camera: str
    from_track_id: int
    to_track_id: int
    label: str
    dominant_color: str
    exit_time: float
    entry_time: float
    delay_seconds: float
    similarity: float
    from_crop: np.ndarray
    to_crop: np.ndarray
    handover_type: str = "sequential"  # 'sequential' (left then appeared) or 'concurrent' (overlapping)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.description:
            if self.handover_type == "sequential":
                self.description = (
                    f"Left {self.from_camera} at {self.exit_time:.1f}s ➔ "
                    f"Transition delay {self.delay_seconds:.1f}s ➔ "
                    f"Appeared on {self.to_camera} at {self.entry_time:.1f}s "
                    f"({self.similarity:.1%} visual match)"
                )
            else:
                self.description = (
                    f"Co-observed on {self.from_camera} ({self.exit_time:.1f}s) and "
                    f"{self.to_camera} ({self.entry_time:.1f}s) "
                    f"({self.similarity:.1%} visual match)"
                )


@dataclass
class CrossCameraJourney:
    """Consolidated multi-camera timeline for a unique vehicle."""

    global_id: int
    label: str
    dominant_color: str
    representative_crop: np.ndarray
    sightings: list[dict[str, Any]]
    handovers: list[CrossCameraHandover] = field(default_factory=list)
    has_handover: bool = False
    total_cameras: int = 1

    @property
    def title(self) -> str:
        color_label = f"{self.dominant_color.capitalize()} {self.label}" if self.dominant_color != "unknown" else self.label.capitalize()
        if self.has_handover:
            cams = " ➔ ".join(s["camera"] for s in self.sightings)
            return f"Vehicle #{self.global_id} ({color_label}) · Journey: {cams}"
        return f"Vehicle #{self.global_id} ({color_label}) · {self.sightings[0]['camera']}"


def track_single_camera(
    video_source: str | Path,
    camera_name: str,
    detector: YOLO,
    processor: AutoProcessor,
    embedder: AutoModel,
    device: str,
    sample_fps: float = 3.0,
    conf_threshold: float = 0.35,
    min_box_area: float = 1200.0,
    time_offset: float = 0.0,
    max_gap_seconds: float = 1.8,
) -> list[SingleCameraTrack]:
    """Process a single camera feed, detect vehicles, track them, and compute SigLIP embeddings."""
    source_str = str(video_source)
    cap = cv2.VideoCapture(source_str)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video stream for camera '{camera_name}': {source_str}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = (total_frames / fps) if total_frames > 0 else 0.0
    step = max(1, int(fps / sample_fps))

    active_tracks: dict[int, dict[str, Any]] = {}
    completed_tracks: list[dict[str, Any]] = []
    track_counter = 1

    frame_idx = 0
    dtype = next(embedder.parameters()).dtype

    try:
        while cap.isOpened():
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            timestamp = (frame_idx / fps) + time_offset
            h_img, w_img = frame.shape[:2]

            res = detector.predict(frame, conf=conf_threshold, verbose=False, device=device)[0]
            boxes = res.boxes

            detections = []
            if boxes is not None:
                for b in boxes:
                    cls_id = int(b.cls[0])
                    cls_name = res.names.get(cls_id, str(cls_id)).lower()
                    if cls_name not in {"car", "truck", "bus", "motorcycle"}:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    conf = float(b.conf[0])
                    box_w = max(1, x2 - x1)
                    box_h = max(1, y2 - y1)
                    if box_w * box_h < min_box_area:
                        continue

                    # Clamp coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    crop_bgr = frame[y1:y2, x1:x2]
                    if crop_bgr.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

                    detections.append({
                        "label": cls_name,
                        "bbox": [x1, y1, x2, y2],
                        "conf": conf,
                        "crop": crop_rgb,
                        "cx": (x1 + x2) / 2.0,
                        "cy": (y1 + y2) / 2.0,
                        "w": box_w,
                        "h": box_h,
                    })

            # Spatial tracking association
            matched_active_ids = set()
            for det in detections:
                best_tid = None
                best_dist = float("inf")

                for tid, trk in active_tracks.items():
                    if tid in matched_active_ids or trk["label"] != det["label"]:
                        continue
                    dx = det["cx"] - trk["cx"]
                    dy = det["cy"] - trk["cy"]
                    dist = math.hypot(dx, dy)
                    diag = math.hypot(trk["w"], trk["h"])
                    max_allowed_dist = max(90.0, diag * 1.3)
                    if dist < max_allowed_dist and dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                if best_tid is not None:
                    matched_active_ids.add(best_tid)
                    trk = active_tracks[best_tid]
                    trk["end_time"] = timestamp
                    trk["cx"] = det["cx"]
                    trk["cy"] = det["cy"]
                    trk["w"] = det["w"]
                    trk["h"] = det["h"]
                    trk["obs_count"] += 1
                    trk["history"].append({
                        "timestamp": timestamp,
                        "center": (det["cx"], det["cy"]),
                        "bbox": det["bbox"],
                    })
                    if det["conf"] > trk["best_conf"]:
                        trk["best_conf"] = det["conf"]
                        trk["best_crop"] = det["crop"]
                        trk["best_time"] = timestamp
                else:
                    tid = track_counter
                    track_counter += 1
                    active_tracks[tid] = {
                        "track_id": tid,
                        "camera": camera_name,
                        "label": det["label"],
                        "start_time": timestamp,
                        "end_time": timestamp,
                        "best_time": timestamp,
                        "best_conf": det["conf"],
                        "best_crop": det["crop"],
                        "cx": det["cx"],
                        "cy": det["cy"],
                        "w": det["w"],
                        "h": det["h"],
                        "obs_count": 1,
                        "start_pos": (det["cx"], det["cy"]),
                        "history": [{
                            "timestamp": timestamp,
                            "center": (det["cx"], det["cy"]),
                            "bbox": det["bbox"],
                        }],
                    }
                    matched_active_ids.add(tid)

            # Check for completed / exited tracks
            inactive_tids = [
                tid for tid, trk in active_tracks.items()
                if tid not in matched_active_ids and (timestamp - trk["end_time"]) > max_gap_seconds
            ]
            for tid in inactive_tids:
                completed_tracks.append(active_tracks.pop(tid))

            frame_idx += step
    finally:
        cap.release()

    # Flush remaining active tracks
    for tid, trk in active_tracks.items():
        completed_tracks.append(trk)

    # Filter spurious detections (require at least 2 observations or strong confidence)
    valid_raw_tracks = [
        t for t in completed_tracks
        if t["obs_count"] >= 2 or t["best_conf"] >= 0.55
    ]

    if not valid_raw_tracks:
        return []

    # Batch embed best crops with SigLIP
    pil_crops = [Image.fromarray(t["best_crop"]) for t in valid_raw_tracks]
    batch_embeddings = []
    batch_size = 16
    for offset in range(0, len(pil_crops), batch_size):
        chunk = pil_crops[offset : offset + batch_size]
        inputs = processor(images=chunk, return_tensors="pt")
        inputs = {
            k: v.to(device=device, dtype=dtype) if k == "pixel_values" else v.to(device)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            feats = embedder.get_image_features(**inputs)
            feats = torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy()
        batch_embeddings.extend(feats)

    results: list[SingleCameraTrack] = []
    for idx, raw in enumerate(valid_raw_tracks):
        # Determine movement and exit status
        history = raw["history"]
        p_start = history[0]["center"]
        p_end = history[-1]["center"]
        displacement = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
        is_moving = displacement > 35.0

        # Did it leave the camera before video ended?
        # A track left the camera if its end_time is notably before the video duration (or near screen border)
        left_cam = False
        if video_duration > 0 and (raw["end_time"] < (video_duration - 1.5)):
            left_cam = True

        entered_cam = (raw["start_time"] - time_offset) > 0.8

        dom_color = extract_dominant_color(raw["best_crop"])

        results.append(
            SingleCameraTrack(
                track_id=raw["track_id"],
                camera_name=camera_name,
                label=raw["label"],
                dominant_color=dom_color,
                start_time=raw["start_time"],
                end_time=raw["end_time"],
                duration=max(0.1, raw["end_time"] - raw["start_time"]),
                best_conf=raw["best_conf"],
                best_time=raw["best_time"],
                best_crop=raw["best_crop"],
                obs_count=raw["obs_count"],
                is_moving=is_moving,
                left_camera=left_cam,
                entered_camera=entered_cam,
                embedding=batch_embeddings[idx],
                trajectory=raw["history"],
            )
        )

    # Sort tracks chronologically
    results.sort(key=lambda t: t.start_time)
    return results


def match_cross_camera_tracks(
    tracks_by_camera: dict[str, list[SingleCameraTrack]],
    similarity_threshold: float = 0.70,
    max_handover_seconds: float = 45.0,
    min_handover_seconds: float = -5.0,
    require_color_match: bool = False,
) -> tuple[list[CrossCameraJourney], list[CrossCameraHandover]]:
    """Associate tracks across cameras to detect vehicle handovers and reconstruct journeys."""
    camera_names = list(tracks_by_camera.keys())
    if len(camera_names) < 2:
        # Single camera fallback: every track is its own journey
        journeys: list[CrossCameraJourney] = []
        gid = 1
        for cam, tracks in tracks_by_camera.items():
            for trk in tracks:
                journeys.append(
                    CrossCameraJourney(
                        global_id=gid,
                        label=trk.label,
                        dominant_color=trk.dominant_color,
                        representative_crop=trk.best_crop,
                        sightings=[{
                            "camera": cam,
                            "track_id": trk.track_id,
                            "start_time": trk.start_time,
                            "end_time": trk.end_time,
                            "crop": trk.best_crop,
                            "is_exit": trk.left_camera,
                            "is_entry": trk.entered_camera,
                        }],
                        handovers=[],
                        has_handover=False,
                        total_cameras=1,
                    )
                )
                gid += 1
        return journeys, []

    # Build candidate cross-camera pairs
    candidate_pairs = []
    for i in range(len(camera_names)):
        cam_a = camera_names[i]
        for j in range(i + 1, len(camera_names)):
            cam_b = camera_names[j]

            for trk_1 in tracks_by_camera[cam_a]:
                for trk_2 in tracks_by_camera[cam_b]:
                    if trk_1.embedding is None or trk_2.embedding is None:
                        continue
                    if trk_1.label != trk_2.label:
                        continue

                    # Calculate visual cosine similarity
                    sim = float(np.dot(trk_1.embedding, trk_2.embedding))
                    if sim < similarity_threshold:
                        continue

                    # Color agreement
                    color_match = (
                        trk_1.dominant_color == trk_2.dominant_color
                        or trk_1.dominant_color in {"unknown", "other"}
                        or trk_2.dominant_color in {"unknown", "other"}
                    )
                    if require_color_match and not color_match:
                        continue

                    # Order chronologically: earlier track is from_cam, later is to_cam
                    if trk_1.start_time <= trk_2.start_time:
                        trk_a, from_cam = trk_1, cam_a
                        trk_b, to_cam = trk_2, cam_b
                    else:
                        trk_a, from_cam = trk_2, cam_b
                        trk_b, to_cam = trk_1, cam_a

                    # Delay from A's end to B's start
                    delay = trk_b.start_time - trk_a.end_time

                    # Is this a sequential handover (A exited, then B appeared)?
                    is_sequential = min_handover_seconds <= delay <= max_handover_seconds
                    # Or overlapping / concurrent observation?
                    is_concurrent = (
                        max(trk_a.start_time, trk_b.start_time)
                        <= min(trk_a.end_time, trk_b.end_time) + 2.0
                    )

                    if not (is_sequential or is_concurrent):
                        continue

                    # Combined score giving preference to higher visual similarity and color match
                    score = sim + (0.05 if color_match else 0.0)
                    handover_type = "sequential" if delay >= 0 else "concurrent"

                    candidate_pairs.append({
                        "from_cam": from_cam,
                        "to_cam": to_cam,
                        "trk_a": trk_a,
                        "trk_b": trk_b,
                        "sim": sim,
                        "score": score,
                        "delay": delay,
                        "type": handover_type,
                    })

    # Sort candidates by match score descending
    candidate_pairs.sort(key=lambda item: item["score"], reverse=True)

    # Disjoint set / union-find to group matched tracklets into global identities
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(key: tuple[str, int]) -> tuple[str, int]:
        if parent.get(key, key) != key:
            parent[key] = find(parent[key])
        return parent.get(key, key)

    def union(k1: tuple[str, int], k2: tuple[str, int]) -> None:
        r1 = find(k1)
        r2 = find(k2)
        if r1 != r2:
            parent[r1] = r2

    used_pairs = set()
    handovers: list[CrossCameraHandover] = []

    for cand in candidate_pairs:
        k_a = (cand["from_cam"], cand["trk_a"].track_id)
        k_b = (cand["to_cam"], cand["trk_b"].track_id)

        # Ensure we don't connect if already connected or if same camera
        root_a = find(k_a)
        root_b = find(k_b)
        if root_a == root_b:
            continue

        # Prevent 1 tracklet from connecting to multiple tracklets on the SAME target camera
        # Check if root_a already has a sighting on to_cam
        group_members = [
            k for k in list(parent.keys()) + [k_a, k_b]
            if find(k) in (root_a, root_b)
        ]
        cams_in_group = [k[0] for k in group_members]
        if cams_in_group.count(cand["to_cam"]) > 1 or cams_in_group.count(cand["from_cam"]) > 1:
            continue

        union(k_a, k_b)
        used_pairs.add((k_a, k_b))

        ho = CrossCameraHandover(
            global_id=0,  # Will be assigned below
            from_camera=cand["from_cam"],
            to_camera=cand["to_cam"],
            from_track_id=cand["trk_a"].track_id,
            to_track_id=cand["trk_b"].track_id,
            label=cand["trk_a"].label,
            dominant_color=cand["trk_a"].dominant_color,
            exit_time=cand["trk_a"].end_time,
            entry_time=cand["trk_b"].start_time,
            delay_seconds=cand["delay"],
            similarity=cand["sim"],
            from_crop=cand["trk_a"].best_crop,
            to_crop=cand["trk_b"].best_crop,
            handover_type=cand["type"],
        )
        handovers.append(ho)

    # Gather all track keys
    all_track_keys = []
    for cam, trks in tracks_by_camera.items():
        for t in trks:
            all_track_keys.append((cam, t.track_id))

    # Group track keys by root
    groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for key in all_track_keys:
        root = find(key)
        groups.setdefault(root, []).append(key)

    # Build journey objects
    journeys: list[CrossCameraJourney] = []
    track_lookup = {
        (cam, t.track_id): t
        for cam, trks in tracks_by_camera.items()
        for t in trks
    }

    # First sort groups: multi-camera groups first, then chronological
    sorted_groups = sorted(
        groups.values(),
        key=lambda members: (
            -len({k[0] for k in members}),  # More distinct cameras first
            min(track_lookup[k].start_time for k in members),
        ),
    )

    global_id_counter = 1
    for members in sorted_groups:
        member_tracks = [track_lookup[k] for k in members]
        member_tracks.sort(key=lambda t: t.start_time)

        # Find matching handovers for this group
        group_keys = set(members)
        group_handovers = [
            h for h in handovers
            if (h.from_camera, h.from_track_id) in group_keys
            and (h.to_camera, h.to_track_id) in group_keys
        ]
        for h in group_handovers:
            h.global_id = global_id_counter

        # Distinct cameras
        unique_cams = len({t.camera_name for t in member_tracks})
        has_ho = unique_cams > 1

        # Pick best representative crop (highest confidence)
        best_overall_trk = max(member_tracks, key=lambda t: t.best_conf)

        sightings = [
            {
                "camera": t.camera_name,
                "track_id": t.track_id,
                "start_time": t.start_time,
                "end_time": t.end_time,
                "crop": t.best_crop,
                "is_exit": t.left_camera,
                "is_entry": t.entered_camera,
                "confidence": t.best_conf,
            }
            for t in member_tracks
        ]

        journeys.append(
            CrossCameraJourney(
                global_id=global_id_counter,
                label=best_overall_trk.label,
                dominant_color=best_overall_trk.dominant_color,
                representative_crop=best_overall_trk.best_crop,
                sightings=sightings,
                handovers=group_handovers,
                has_handover=has_ho,
                total_cameras=unique_cams,
            )
        )
        global_id_counter += 1

    return journeys, handovers


def analyze_multiple_cameras(
    camera_sources: list[dict[str, Any]],
    detector: YOLO | None = None,
    processor: AutoProcessor | None = None,
    embedder: AutoModel | None = None,
    device: str | None = None,
    sample_fps: float = 3.0,
    similarity_threshold: float = 0.70,
    max_handover_seconds: float = 45.0,
    min_handover_seconds: float = -5.0,
    conf_threshold: float = 0.35,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Ingest and track multiple camera video feeds, matching vehicles across cameras.

    camera_sources: list of dicts:
      [
        {"name": "Gate Cam", "source": "data/videos/gate_cam.mp4", "time_offset": 0.0},
        {"name": "Rear Cam", "source": "data/videos/rear_cam.mp4", "time_offset": 0.0},
      ]
    """
    if len(camera_sources) < 1:
        raise ValueError("At least one camera source must be provided.")

    if detector is None or processor is None or embedder is None or device is None:
        detector, processor, embedder, device = load_visual_models()

    tracks_by_camera: dict[str, list[SingleCameraTrack]] = {}
    cameras_meta: list[dict[str, Any]] = []

    total_cams = len(camera_sources)
    for idx, cam_info in enumerate(camera_sources):
        name = str(cam_info.get("name", f"Camera {idx + 1}"))
        source = cam_info["source"]
        offset = float(cam_info.get("time_offset", 0.0))

        if progress_callback:
            progress_callback(idx / total_cams, f"Tracking camera {idx + 1}/{total_cams}: {name}...")

        tracks = track_single_camera(
            video_source=source,
            camera_name=name,
            detector=detector,
            processor=processor,
            embedder=embedder,
            device=device,
            sample_fps=sample_fps,
            conf_threshold=conf_threshold,
            time_offset=offset,
        )
        tracks_by_camera[name] = tracks
        cameras_meta.append({
            "name": name,
            "source": str(source),
            "offset": offset,
            "track_count": len(tracks),
        })

    if progress_callback:
        progress_callback(0.95, "Matching vehicle identities across cameras...")

    journeys, handovers = match_cross_camera_tracks(
        tracks_by_camera=tracks_by_camera,
        similarity_threshold=similarity_threshold,
        max_handover_seconds=max_handover_seconds,
        min_handover_seconds=min_handover_seconds,
    )

    if progress_callback:
        progress_callback(1.0, "Analysis complete!")

    # Summary statistics
    multi_cam_journeys = [j for j in journeys if j.has_handover]
    avg_delay = (
        float(np.mean([h.delay_seconds for h in handovers if h.delay_seconds >= 0]))
        if any(h.delay_seconds >= 0 for h in handovers)
        else 0.0
    )

    return {
        "cameras": cameras_meta,
        "tracks_by_camera": tracks_by_camera,
        "journeys": journeys,
        "handovers": handovers,
        "multi_camera_vehicles_count": len(multi_cam_journeys),
        "total_vehicles_count": len(journeys),
        "average_handover_delay": avg_delay,
    }
