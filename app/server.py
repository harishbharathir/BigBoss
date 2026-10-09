"""FastAPI backend server for BiggBoss Multi-Stream Video Intelligence.

Serves the modern enterprise HTML/CSS/JS dashboard and exposes REST APIs
for all AI capabilities without modifying the core models:
- YOLO-World + SigLIP detection & embedding
- Cross-camera vehicle Re-ID & handover tracking
- Grounded conversational chatbot (Fast Local Agent + GPT4All GGUF)
- Clarify-once persistent spatial memory
- Single-stream open-vocabulary semantic search
- Automated standing queries & alert evaluation
- On-premise privacy redaction filter
- Research benchmark ablation matrix
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from app.alerts.rules import AlertManager
from app.config import load_config
from app.eval.ablation import detailed_ablation_matrix, run_empirical_benchmark
from app.privacy.redaction import PrivacyFilter
from app.reid.cross_camera import (
    CrossCameraHandover,
    CrossCameraJourney,
    SingleCameraTrack,
    analyze_multiple_cameras,
)
from app.retrieval.chatbot import ChatbotMessage, VideoIntelligenceChatbot
from app.retrieval.conversational_engine import (
    ConversationalResult,
    GroundedMatch,
    MultiStreamConversationalEngine,
    format_timestamp,
)
from app.retrieval.visual_search import analyze_video, load_visual_models, search_frames
from app.ui.clarify_flow import ClarifySession

logger = logging.getLogger("biggboss.server")
logging.basicConfig(level=logging.INFO)

# Config and directories
config = load_config()
project_root: Path = config["_project_root"]
video_dir: Path = config["_paths"]["videos"]
video_dir.mkdir(parents=True, exist_ok=True)
static_dir = project_root / "app" / "static"
static_dir.mkdir(parents=True, exist_ok=True)

# Application state singleton
class AppState:
    def __init__(self):
        self.visual_models: tuple[Any, Any, Any, str] | None = None
        self.multicam_results: dict[str, Any] | None = None
        self.multicam_config: dict[str, Any] | None = None
        self.clarify_session = ClarifySession()
        self.alert_manager = AlertManager()
        self.privacy_filter = PrivacyFilter()
        self.privacy_mode_enabled: bool = False
        self.single_analysis: dict[str, Any] | None = None
        self.chat_messages: list[dict[str, Any]] = []
        self.gpt4all_bot: VideoIntelligenceChatbot | None = None
        self.loaded_gpt4all_model: str | None = None

    def get_models(self) -> tuple[Any, Any, Any, str]:
        if self.visual_models is None:
            logger.info("Loading visual models (YOLO-World + SigLIP)...")
            self.visual_models = load_visual_models()
            logger.info(f"Visual models loaded on device: {self.visual_models[3]}")
        return self.visual_models

state = AppState()

# Helpers for image serialization
def ndarray_to_base64(img: np.ndarray | None, redact: bool = False) -> str:
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return ""
    try:
        proc_img = img.copy()
        if redact:
            proc_img = state.privacy_filter.redact_crop(proc_img)

        if len(proc_img.shape) == 2:
            proc_img = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2RGB)
        elif proc_img.shape[2] == 4:
            proc_img = cv2.cvtColor(proc_img, cv2.COLOR_RGBA2RGB)

        pil_img = Image.fromarray(proc_img)
        # Limit dimension for crisp, ultra-fast transmission
        max_dim = 640
        w, h = pil_img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    except Exception as e:
        logger.warning(f"Error serializing image: {e}")
        return ""

# Initialize FastAPI
app = FastAPI(
    title="BiggBoss CCTV Intelligence API",
    description="Multi-stream video intelligence, Re-ID, and grounded conversational search.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount video directory for direct HTML5 video playback
app.mount("/media/videos", StaticFiles(directory=str(video_dir)), name="videos")

# Mount static files (CSS, JS, Assets)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# =========================================================================
# SYSTEM & HEALTH ENDPOINTS
# =========================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = static_dir / "index.html"
    if not index_file.is_file():
        return HTMLResponse("<h1>BiggBoss UI is building...</h1>", status_code=503)
    return FileResponse(str(index_file))


@app.get("/api/status")
async def get_system_status():
    models_ready = state.visual_models is not None
    device_name = state.visual_models[3] if models_ready else "Not Loaded"
    has_results = state.multicam_results is not None

    stats = {
        "models_ready": models_ready,
        "device": device_name,
        "indexed": has_results,
        "camera_count": len(state.multicam_results["cameras"]) if has_results else 0,
        "entity_count": (
            state.multicam_results.get("total_entities_count", state.multicam_results.get("total_vehicles_count", 0))
            if has_results else 0
        ),
        "handover_count": len(state.multicam_results["handovers"]) if has_results else 0,
        "avg_delay": state.multicam_results.get("average_handover_delay", 0.0) if has_results else 0.0,
        "privacy_enabled": state.privacy_mode_enabled,
        "kb_count": len(state.clarify_session.get_mappings()),
        "gpt4all_loaded": state.loaded_gpt4all_model if (state.gpt4all_bot and state.gpt4all_bot.is_gpt4all_loaded) else None,
    }
    return JSONResponse(stats)


@app.post("/api/models/load")
async def preload_models():
    try:
        models = state.get_models()
        return JSONResponse({"status": "success", "device": models[3]})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# =========================================================================
# VIDEO MANAGEMENT & INGESTION ENDPOINTS
# =========================================================================

@app.get("/api/videos")
async def list_available_videos():
    videos = []
    if video_dir.exists():
        for p in sorted(video_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
                videos.append({
                    "name": p.name,
                    "path": str(p.resolve()),
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                    "url": f"/media/videos/{p.name}",
                })
    return JSONResponse({"videos": videos})


@app.post("/api/videos/upload")
async def upload_videos(files: list[UploadFile] = File(...)):
    saved = []
    for file in files:
        target_path = video_dir / file.filename
        contents = await file.read()
        target_path.write_bytes(contents)
        saved.append({
            "name": file.filename,
            "path": str(target_path.resolve()),
            "size_mb": round(len(contents) / (1024 * 1024), 2),
            "url": f"/media/videos/{file.filename}",
        })
    return JSONResponse({"uploaded": saved})


class IngestConfigRequest(BaseModel):
    sources: list[dict[str, Any]]
    sample_fps: float = 2.5
    similarity_threshold: float = 0.70
    max_handover_seconds: float = 45.0


@app.post("/api/ingest")
async def ingest_cameras(req: IngestConfigRequest):
    if not req.sources:
        raise HTTPException(status_code=400, detail="No camera sources provided.")

    try:
        models = state.get_models()
        results = analyze_multiple_cameras(
            camera_sources=req.sources,
            detector=models[0],
            processor=models[1],
            embedder=models[2],
            device=models[3],
            sample_fps=req.sample_fps,
            similarity_threshold=req.similarity_threshold,
            max_handover_seconds=req.max_handover_seconds,
        )
        state.multicam_results = results
        state.multicam_config = {
            "cams": [c["name"] for c in req.sources],
            "sim_threshold": req.similarity_threshold,
        }

        return JSONResponse({
            "status": "success",
            "camera_count": len(results["cameras"]),
            "vehicles_count": results["total_vehicles_count"],
            "handovers_count": len(results["handovers"]),
            "avg_delay": results.get("average_handover_delay", 0.0),
        })
    except Exception as exc:
        logger.error(f"Ingest failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/ingest/quick")
async def quick_index_project_feeds():
    """Quick 1-click index of gate_cam.mp4 and rear_cam.mp4."""
    gate_file = video_dir / "gate_cam.mp4"
    rear_file = video_dir / "rear_cam.mp4"
    if not gate_file.exists() or not rear_file.exists():
        raise HTTPException(status_code=404, detail="Default gate_cam.mp4 and rear_cam.mp4 not found in data/videos.")

    try:
        models = state.get_models()
        sources = [
            {"name": "Gate Cam", "source": str(gate_file.resolve()), "time_offset": 0.0},
            {"name": "Rear Cam", "source": str(rear_file.resolve()), "time_offset": 0.0},
        ]
        results = analyze_multiple_cameras(
            camera_sources=sources,
            detector=models[0],
            processor=models[1],
            embedder=models[2],
            device=models[3],
            sample_fps=2.5,
            similarity_threshold=0.70,
            max_handover_seconds=45.0,
        )
        state.multicam_results = results
        state.multicam_config = {"cams": ["Gate Cam", "Rear Cam"], "sim_threshold": 0.70}

        return JSONResponse({
            "status": "success",
            "camera_count": len(results["cameras"]),
            "vehicles_count": results["total_vehicles_count"],
            "handovers_count": len(results["handovers"]),
            "avg_delay": results.get("average_handover_delay", 0.0),
        })
    except Exception as exc:
        logger.error(f"Quick index failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/ingest/results")
async def get_ingest_results():
    if state.multicam_results is None:
        return JSONResponse({"indexed": False})

    res = state.multicam_results
    redact = state.privacy_mode_enabled

    # Serialize handovers
    serialized_handovers = []
    for ho in res.get("handovers", []):
        serialized_handovers.append({
            "global_id": ho.global_id,
            "label": ho.label,
            "dominant_color": ho.dominant_color,
            "from_camera": ho.from_camera,
            "to_camera": ho.to_camera,
            "exit_time": round(ho.exit_time, 2),
            "entry_time": round(ho.entry_time, 2),
            "delay_seconds": round(ho.delay_seconds, 2),
            "similarity": round(ho.similarity, 4),
            "handover_type": ho.handover_type,
            "description": ho.description,
            "from_crop": ndarray_to_base64(ho.from_crop, redact=redact),
            "to_crop": ndarray_to_base64(ho.to_crop, redact=redact),
        })

    # Serialize journeys
    serialized_journeys = []
    for j in res.get("journeys", []):
        sightings_data = []
        for s in j.sightings:
            sightings_data.append({
                "camera": s.get("camera"),
                "start_time": round(s.get("start_time", 0.0), 2),
                "end_time": round(s.get("end_time", 0.0), 2),
                "crop": ndarray_to_base64(s.get("crop"), redact=redact),
            })

        serialized_journeys.append({
            "global_id": j.global_id,
            "title": j.title,
            "label": j.label,
            "dominant_color": j.dominant_color,
            "has_handover": j.has_handover,
            "camera_path_str": " ➔ ".join(s.get("camera", "") for s in j.sightings),
            "representative_crop": ndarray_to_base64(j.representative_crop, redact=redact),
            "sightings": sightings_data,
        })

    return JSONResponse({
        "indexed": True,
        "cameras": res.get("cameras", []),
        "total_vehicles_count": res.get("total_vehicles_count", 0),
        "total_entities_count": res.get("total_entities_count", res.get("total_vehicles_count", 0)),
        "average_handover_delay": round(res.get("average_handover_delay", 0.0), 2),
        "handovers": serialized_handovers,
        "journeys": serialized_journeys,
    })


# =========================================================================
# CHATBOT & CONVERSATIONAL GROUNDING ENDPOINTS
# =========================================================================

class ChatQueryRequest(BaseModel):
    query: str
    use_gpt4all: bool = False
    gpt4all_model: str = "Llama-3.2-1B-Instruct-Q4_0.gguf"


@app.post("/api/chat")
async def chat_interaction(req: ChatQueryRequest):
    if state.multicam_results is None:
        raise HTTPException(
            status_code=400,
            detail="CCTV streams are not indexed yet. Please ingest camera feeds first.",
        )

    clean_query = req.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Record user message in history
    state.chat_messages.append({"role": "user", "content": clean_query})

    models = state.get_models()
    engine = MultiStreamConversationalEngine(
        processor=models[1],
        embedder=models[2],
        device=models[3],
        clarify_session=state.clarify_session,
    )

    # Initialize bot
    if state.gpt4all_bot is None or req.use_gpt4all:
        if state.gpt4all_bot is None:
            state.gpt4all_bot = VideoIntelligenceChatbot(engine=engine, gpt4all_model_name=req.gpt4all_model)
        else:
            state.gpt4all_bot.engine = engine

    bot = state.gpt4all_bot

    reply: ChatbotMessage = bot.chat(
        user_query=clean_query,
        multicam_results=state.multicam_results,
        use_gpt4all=req.use_gpt4all,
    )

    redact = state.privacy_mode_enabled

    # Clarification handling
    if reply.needs_clarification:
        response_data = {
            "role": "assistant",
            "content": reply.content,
            "needs_clarification": True,
            "clarification_entity": reply.clarification_entity,
            "clarification_prompt": reply.clarification_prompt,
            "available_cameras": [c["name"] for c in state.multicam_results["cameras"]],
            "model_used": reply.model_used,
        }
        return JSONResponse(response_data)

    # Serialize primary match
    primary_data = None
    if reply.primary_match:
        pm = reply.primary_match
        primary_data = {
            "camera_name": pm.camera_name,
            "timestamp": round(pm.timestamp, 2),
            "timestamp_str": pm.timestamp_str,
            "confidence": round(pm.confidence, 4),
            "label": pm.label,
            "dominant_color": pm.dominant_color,
            "event_type": pm.event_type,
            "details": pm.details,
            "crop": ndarray_to_base64(pm.crop, redact=redact),
            "source_video": getattr(pm, "source_video", ""),
            "clip_url": getattr(pm, "clip_url", ""),
            "start_pts": round(getattr(pm, "start_pts", 0.0), 2),
            "end_pts": round(getattr(pm, "end_pts", 0.0), 2),
        }

    # Serialize timeline steps
    timeline_data = []
    for step in reply.timeline_steps:
        timeline_data.append({
            "camera": step.get("camera"),
            "start_time": round(step.get("start_time", 0.0), 2),
            "end_time": round(step.get("end_time", 0.0), 2),
            "status": step.get("status"),
            "crop": ndarray_to_base64(step.get("crop"), redact=redact),
        })

    assistant_msg = {
        "role": "assistant",
        "content": reply.content,
        "primary_match": primary_data,
        "timeline_steps": timeline_data,
        "model_used": reply.model_used,
        "needs_clarification": False,
    }
    state.chat_messages.append(assistant_msg)

    return JSONResponse(assistant_msg)


@app.get("/api/chat/history")
async def get_chat_history():
    return JSONResponse({"history": state.chat_messages})


@app.post("/api/chat/clear")
async def clear_chat_history():
    state.chat_messages = []
    return JSONResponse({"status": "cleared"})


# =========================================================================
# GPT4ALL CONFIGURATION ENDPOINTS
# =========================================================================

@app.get("/api/gpt4all/status")
async def get_gpt4all_status():
    available_models = [
        "Llama-3.2-1B-Instruct-Q4_0.gguf",
        "all-MiniLM-L6-v2-f16.gguf",
        "qwen2.5-coder-7b-instruct-q4_0.gguf",
    ]
    cache_dir = Path(os.environ.get("USERPROFILE", "")) / ".cache" / "gpt4all"

    model_info = []
    for m in available_models:
        f = cache_dir / m
        is_cached = f.exists()
        size_mb = round(f.stat().st_size / (1024 * 1024), 1) if is_cached else 0
        model_info.append({
            "name": m,
            "cached": is_cached,
            "size_mb": size_mb,
            "is_loaded": state.loaded_gpt4all_model == m and (state.gpt4all_bot and state.gpt4all_bot.is_gpt4all_loaded),
        })

    return JSONResponse({
        "installed": True,
        "active_model": state.loaded_gpt4all_model,
        "models": model_info,
    })


@app.post("/api/gpt4all/load")
async def load_gpt4all_model(req: dict[str, str]):
    model_name = req.get("model_name", "Llama-3.2-1B-Instruct-Q4_0.gguf")
    models = state.get_models()
    engine = MultiStreamConversationalEngine(models[1], models[2], models[3], state.clarify_session)
    bot = VideoIntelligenceChatbot(engine=engine, gpt4all_model_name=model_name)
    ok = bot.load_gpt4all(model_name)
    if ok:
        state.gpt4all_bot = bot
        state.loaded_gpt4all_model = model_name
        return JSONResponse({"status": "success", "loaded_model": model_name})
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load GPT4All model {model_name}")


# =========================================================================
# CLARIFY-ONCE PERSISTENT KNOWLEDGE BASE ENDPOINTS
# =========================================================================

@app.get("/api/kb")
async def get_knowledge_base():
    return JSONResponse({"mappings": state.clarify_session.get_mappings()})


class KBMappingRequest(BaseModel):
    referent: str
    target: str


@app.post("/api/kb/learn")
async def learn_kb_mapping(req: KBMappingRequest):
    if not req.referent.strip() or not req.target.strip():
        raise HTTPException(status_code=400, detail="Referent and Target are required.")
    state.clarify_session.learn_mapping(req.referent, req.target)
    return JSONResponse({"status": "success", "mappings": state.clarify_session.get_mappings()})


@app.delete("/api/kb/{referent}")
async def delete_kb_mapping(referent: str):
    ok = state.clarify_session.delete_mapping(referent)
    return JSONResponse({"status": "deleted" if ok else "not_found", "mappings": state.clarify_session.get_mappings()})


# =========================================================================
# SINGLE-STREAM SEMANTIC SEARCH ENDPOINTS
# =========================================================================

class SingleIndexRequest(BaseModel):
    video_path: str
    max_samples: int = 32


@app.post("/api/search/index")
async def index_single_clip(req: SingleIndexRequest):
    try:
        models = state.get_models()
        analysis = analyze_video(
            video_path=req.video_path,
            max_samples=req.max_samples,
            detector=models[0],
            processor=models[1],
            embedder=models[2],
            device=models[3],
        )
        state.single_analysis = analysis
        return JSONResponse({
            "status": "success",
            "video": req.video_path,
            "sample_count": analysis["sample_count"],
            "duration": round(analysis["duration_seconds"], 2),
            "fps": round(analysis["fps"], 2),
        })
    except Exception as exc:
        logger.error(f"Single analysis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


class SingleSearchRequest(BaseModel):
    query: str
    top_k: int = 6


@app.post("/api/search/query")
async def search_single_clip(req: SingleSearchRequest):
    if state.single_analysis is None:
        raise HTTPException(status_code=400, detail="No single video clip indexed yet.")
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    models = state.get_models()
    matches = search_frames(state.single_analysis, query, models[1], models[2], models[3])[:req.top_k]

    serialized_matches = []
    redact = state.privacy_mode_enabled
    for m in matches:
        serialized_matches.append({
            "timestamp": round(m["timestamp"], 2),
            "timestamp_str": format_timestamp(m["timestamp"]),
            "score": round(m["score"], 4),
            "frame_index": m["frame_index"],
            "image": ndarray_to_base64(m["image"], redact=redact),
            "boxes": m.get("boxes", []),
        })

    return JSONResponse({"query": query, "matches": serialized_matches})


# =========================================================================
# STANDING RULES & REAL-TIME ALERTS ENDPOINTS
# =========================================================================

@app.get("/api/alerts/rules")
async def get_standing_rules():
    return JSONResponse({
        "rules": state.alert_manager.standing_rules,
        "history_count": len(state.alert_manager.history),
    })


class AddRuleRequest(BaseModel):
    name: str
    condition: str
    threshold: float = 0.75
    camera: str = "all"


@app.post("/api/alerts/rules")
async def add_standing_rule(req: AddRuleRequest):
    rule = state.alert_manager.add_standing_rule(
        name=req.name,
        condition=req.condition,
        threshold=req.threshold,
        camera=req.camera,
    )
    return JSONResponse({"status": "success", "rule": rule, "rules": state.alert_manager.standing_rules})


@app.post("/api/alerts/rules/{rule_id}/toggle")
async def toggle_standing_rule(rule_id: str):
    for r in state.alert_manager.standing_rules:
        if r["id"] == rule_id:
            r["enabled"] = not r.get("enabled", True)
            return JSONResponse({"status": "toggled", "rule": r, "rules": state.alert_manager.standing_rules})
    raise HTTPException(status_code=404, detail="Rule not found.")


@app.get("/api/alerts/evaluate")
async def evaluate_alerts():
    if state.multicam_results is None:
        return JSONResponse({"evaluated": False, "alerts": []})

    all_trks = [t for trks in state.multicam_results["tracks_by_camera"].values() for t in trks]
    matches_to_eval = [
        GroundedMatch(
            camera_name=t.camera_name,
            timestamp=t.end_time,
            timestamp_str=format_timestamp(t.end_time),
            crop=t.best_crop,
            confidence=t.best_conf,
            label=t.label,
            track_id=t.track_id,
            event_type="left" if t.left_camera else "sighted",
            dominant_color=t.dominant_color,
        )
        for t in all_trks
    ]
    triggered = state.alert_manager.evaluate_multi_stream(matches_to_eval)
    redact = state.privacy_mode_enabled

    serialized = []
    for alt in triggered:
        serialized.append({
            "rule_name": alt["rule_name"],
            "camera": alt["camera"],
            "timestamp": round(alt["timestamp"], 2),
            "timestamp_str": alt["timestamp_str"],
            "score": round(alt["score"], 4),
            "event_type": alt["event_type"],
            "label": alt["label"],
            "crop": ndarray_to_base64(alt.get("crop"), redact=redact),
        })

    return JSONResponse({"evaluated": True, "count": len(serialized), "alerts": serialized})


# =========================================================================
# ON-PREM PRIVACY REDACTION ENDPOINTS
# =========================================================================

@app.get("/api/privacy")
async def get_privacy_status():
    return JSONResponse({"enabled": state.privacy_mode_enabled})


@app.post("/api/privacy/toggle")
async def toggle_privacy():
    state.privacy_mode_enabled = not state.privacy_mode_enabled
    return JSONResponse({"enabled": state.privacy_mode_enabled})


@app.get("/api/privacy/comparison")
async def get_privacy_comparison():
    """Returns side-by-side comparison of raw vs redacted crop from current index."""
    if state.multicam_results and state.multicam_results.get("handovers"):
        sample_ho = state.multicam_results["handovers"][0]
        raw_crop = sample_ho.from_crop
        redacted_crop = state.privacy_filter.redact_crop(raw_crop, blur_plate=True)
        return JSONResponse({
            "available": True,
            "raw": ndarray_to_base64(raw_crop, redact=False),
            "redacted": ndarray_to_base64(redacted_crop, redact=False),
            "camera": sample_ho.from_camera,
            "timestamp": round(sample_ho.exit_time, 2),
        })
    return JSONResponse({"available": False})


# =========================================================================
# RESEARCH BENCHMARK & ABLATION MATRIX ENDPOINTS
# =========================================================================

@app.get("/api/research/ablation")
async def get_ablation_matrix():
    data = detailed_ablation_matrix()
    return JSONResponse({
        "status": "SIMULATED_FIXTURE",
        "notice": "Synthetic architecture baseline fixture. Run live benchmark for validated local hardware metrics.",
        "matrix": data,
        "kpis": {
            "retrieval_map": "0.852 [Target]",
            "retrieval_map_delta": "+3.2% vs Baseline",
            "grounding_accuracy": "94.0% [Target]",
            "grounding_accuracy_delta": "+18.0% vs Baseline",
            "query_latency": "68.4 ms [Target]",
            "query_latency_delta": "-52% (2.08x faster)",
        },
    })


@app.post("/api/research/benchmark/run")
async def execute_live_benchmark():
    """Run real empirical evaluation measuring latency, grounding, and abstention."""
    models = state.visual_models
    results = run_empirical_benchmark(models=models)
    return JSONResponse(results)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
