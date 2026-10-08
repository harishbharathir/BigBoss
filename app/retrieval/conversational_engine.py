"""Multi-Stream Conversational Intelligence Engine.

Resolves natural language queries (e.g., 'when did the yellow car left',
'did a red car pass through the main gate in the last hour?') into grounded,
localized answers: specific camera + timestamp + visual evidence crop + timeline.

Integrates:
- Open-vocabulary SigLIP vision-language search
- Clarify-once persistent spatial memory (saved to data/knowledge_base.json)
- Cross-camera continuity & timeline reconstruction
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.reid.cross_camera import (
    CrossCameraHandover,
    CrossCameraJourney,
    SingleCameraTrack,
)
from app.ui.clarify_flow import ClarifySession


COLOR_KEYWORDS = {
    "yellow": ["yellow", "gold"],
    "red": ["red", "crimson"],
    "blue": ["blue", "navy", "cyan"],
    "green": ["green"],
    "black": ["black", "dark"],
    "white": ["white"],
    "silver": ["silver", "gray", "grey"],
    "orange": ["orange"],
}

KNOWN_SPATIAL_REFERENTS = [
    "main gate",
    "gate cam",
    "rear cam",
    "gate",
    "rear exit",
    "back gate",
    "parking lot",
    "lobby",
    "entrance",
    "exit",
    "front cam",
    "north gate",
    "south gate",
]


@dataclass
class GroundedMatch:
    """A grounded visual match resolved to camera + timestamp + crop."""

    camera_name: str
    timestamp: float
    timestamp_str: str
    crop: np.ndarray
    confidence: float
    label: str
    track_id: int
    event_type: str  # 'left' (exit), 'entered', 'sighted', 'handover'
    dominant_color: str
    details: str = ""
    frame_image: np.ndarray | None = None
    journey: CrossCameraJourney | None = None
    handover: CrossCameraHandover | None = None


@dataclass
class ConversationalResult:
    """Complete grounded answer to a conversational query."""

    query: str
    resolved_query: str
    answer_text: str
    grounded_matches: list[GroundedMatch]
    primary_match: GroundedMatch | None = None
    needs_clarification: bool = False
    clarification_entity: str | None = None
    clarification_prompt: str | None = None
    reconstructed_timeline: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS.ss."""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def extract_query_intent(query: str) -> dict[str, Any]:
    """Parse query for target entity, colors, clothing attributes, spatial referents, and action."""
    q_lower = query.lower().strip()

    # Detect action
    is_exit = bool(re.search(r"\b(left|leave|leaving|exit|exited|departed|depart)\b", q_lower))
    is_entry = bool(re.search(r"\b(entered|enter|arrived|came|entry)\b", q_lower))
    is_journey = bool(
        re.search(r"\b(where did|path|trajectory|timeline|handover|cross|journey|route|track across)\b", q_lower)
    )

    # Detect colors
    detected_color = None
    for color, synonyms in COLOR_KEYWORDS.items():
        if any(re.search(r"\b" + re.escape(syn) + r"\b", q_lower) for syn in synonyms):
            detected_color = color
            break

    # Detect person and clothing indicators
    person_patterns = [
        r"\b(who|someone|somebody|pedestrian|man|woman|guy|boy|girl|individual|people|person|anyone|anybody)\b",
        r"\b(wearing|dressed in|dressed|carrying|shirt|t-shirt|tshirt|jacket|hoodie|coat|pants|trousers|jeans|hat|cap)\b",
    ]
    is_person_query = any(re.search(pat, q_lower) for pat in person_patterns)

    # Detect explicit vehicle classes
    explicit_vehicle = None
    for obj in ["car", "truck", "bus", "motorcycle", "sedan", "suv", "van", "bicycle"]:
        if re.search(r"\b" + re.escape(obj) + r"\b", q_lower):
            explicit_vehicle = obj
            break

    if is_person_query and not explicit_vehicle:
        target_object = "person"
    elif explicit_vehicle:
        target_object = explicit_vehicle
    else:
        target_object = "vehicle"

    # Detect clothing item descriptor (e.g. "wearing blue shirt", "blue shirt")
    clothing_match = re.search(r"\b(wearing|dressed in|in)\s+([a-z\s]+?\b(shirt|t-shirt|tshirt|jacket|hoodie|coat|pants|jeans|dress|hat|cap)\b)", q_lower)
    clothing_item = clothing_match.group(2).strip() if clothing_match else None

    # Build semantic search phrase for vision-language embedder
    if target_object == "person":
        if clothing_item:
            semantic_phrase = f"person wearing {clothing_item}"
        elif detected_color and any(w in q_lower for w in ["shirt", "jacket", "hoodie", "coat", "clothes", "dress"]):
            semantic_phrase = f"person wearing {detected_color} shirt"
        elif detected_color:
            semantic_phrase = f"person wearing {detected_color}"
        else:
            semantic_phrase = "person walking"
    elif detected_color and target_object:
        semantic_phrase = f"{detected_color} {target_object}"
    elif detected_color:
        semantic_phrase = f"{detected_color} vehicle"
    elif target_object:
        semantic_phrase = target_object
    else:
        semantic_phrase = query

    return {
        "is_exit": is_exit,
        "is_entry": is_entry,
        "is_journey": is_journey,
        "detected_color": detected_color,
        "target_object": target_object,
        "clothing_item": clothing_item,
        "semantic_phrase": semantic_phrase,
    }


def find_unresolved_referent(
    query: str,
    clarify_session: ClarifySession,
    active_cameras: list[str],
) -> tuple[str | None, str | None]:
    """Check if query mentions a spatial referent that is not yet mapped in knowledge base."""
    q_lower = query.lower()
    cam_names_lower = [c.lower() for c in active_cameras]

    # Check known referents
    for referent in KNOWN_SPATIAL_REFERENTS:
        pattern = r"\b" + re.escape(referent) + r"\b"
        if re.search(pattern, q_lower):
            # Check if this exact referent is already in knowledge base
            if referent.lower() in clarify_session.kb:
                continue

            # Check if it already directly matches an active camera name
            direct_match = any(referent.lower() in c or c in referent.lower() for c in cam_names_lower)
            if direct_match and len(active_cameras) > 0:
                continue

            # Needs clarification
            prompt = (
                f"Which camera corresponds to '**{referent}**'? "
                f"Please select the camera feed to remember this location permanently."
            )
            return referent, prompt

    return None, None


def resolve_camera_scope(
    query: str,
    clarify_session: ClarifySession,
    active_cameras: list[str],
) -> str | None:
    """Resolve which specific camera the query is targeting, if any."""
    q_lower = query.lower()

    # Check KB substitutions first
    for entity, mapping in clarify_session.kb.items():
        if re.search(r"\b" + re.escape(entity) + r"\b", q_lower):
            for cam in active_cameras:
                if cam.lower() in mapping.lower() or mapping.lower() in cam.lower():
                    return cam

    # Check direct camera mentions
    for cam in active_cameras:
        if cam.lower() in q_lower:
            return cam
        # E.g. "gate" in "gate cam"
        short_name = cam.lower().replace("cam", "").replace("camera", "").strip()
        if short_name and re.search(r"\b" + re.escape(short_name) + r"\b", q_lower):
            return cam

    return None


class MultiStreamConversationalEngine:
    """Conversational question-answering over multi-camera video streams."""

    def __init__(
        self,
        processor: AutoProcessor,
        embedder: AutoModel,
        device: str,
        clarify_session: ClarifySession | None = None,
    ) -> None:
        self.processor = processor
        self.embedder = embedder
        self.device = device
        self.clarify_session = clarify_session or ClarifySession()

    def embed_text(self, text: str) -> np.ndarray:
        """Compute normalized SigLIP embedding for a text phrase."""
        inputs = self.processor(text=[text], padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            vec = self.embedder.get_text_features(**inputs)
            vec = torch.nn.functional.normalize(vec.float(), dim=-1).cpu().numpy()[0]
        return vec

    def answer_query(
        self,
        query: str,
        multicam_results: dict[str, Any] | None,
        forced_camera: str | None = None,
    ) -> ConversationalResult:
        """Answer a natural-language query using indexed multi-camera data."""
        clean_query = query.strip()
        if not clean_query:
            return ConversationalResult(
                query="",
                resolved_query="",
                answer_text="Please enter a question about your CCTV footage (e.g., 'when did the yellow car left').",
                grounded_matches=[],
            )

        if not multicam_results or not multicam_results.get("tracks_by_camera"):
            return ConversationalResult(
                query=clean_query,
                resolved_query=clean_query,
                answer_text=(
                    "⚠️ **No multi-camera streams have been indexed yet.**\n\n"
                    "Please select CCTV footage files in the sidebar and click "
                    "**'⚡ Ingest & Analyze Cameras'** to index all video streams."
                ),
                grounded_matches=[],
            )

        tracks_by_camera: dict[str, list[SingleCameraTrack]] = multicam_results["tracks_by_camera"]
        handovers: list[CrossCameraHandover] = multicam_results.get("handovers", [])
        journeys: list[CrossCameraJourney] = multicam_results.get("journeys", [])
        active_cams = list(tracks_by_camera.keys())

        # Check for Clarify-Once spatial referent
        unresolved_entity, clar_prompt = find_unresolved_referent(
            clean_query, self.clarify_session, active_cams
        )
        if unresolved_entity and not forced_camera:
            return ConversationalResult(
                query=clean_query,
                resolved_query=clean_query,
                answer_text=f"❓ **Clarification needed:** {clar_prompt}",
                grounded_matches=[],
                needs_clarification=True,
                clarification_entity=unresolved_entity,
                clarification_prompt=clar_prompt,
            )

        # Resolve camera scope from knowledge base
        resolved_camera = forced_camera or resolve_camera_scope(
            clean_query, self.clarify_session, active_cams
        )

        intent = extract_query_intent(clean_query)
        semantic_phrase = intent["semantic_phrase"]

        # Embed semantic phrase
        text_vec = self.embed_text(semantic_phrase)

        # Also embed auxiliary variants for robust matching
        aux_text = [clean_query]
        if intent["target_object"] == "person":
            if intent["detected_color"]:
                aux_text.append(f"person wearing {intent['detected_color']}")
                aux_text.append(f"a person in {intent['detected_color']} clothes")
                aux_text.append(f"person wearing a {intent['detected_color']} shirt")
            else:
                aux_text.append("a person walking")
                aux_text.append("pedestrian")
        elif intent["detected_color"]:
            aux_text.append(f"a {intent['detected_color']} vehicle")
            aux_text.append(f"{intent['detected_color']} car")

        aux_vecs = [self.embed_text(p) for p in aux_text] if aux_text else []

        # Candidate collection across cameras
        candidates: list[GroundedMatch] = []

        cams_to_search = [resolved_camera] if resolved_camera and resolved_camera in tracks_by_camera else active_cams

        for cam_name in cams_to_search:
            cam_tracks = tracks_by_camera[cam_name]
            for trk in cam_tracks:
                if trk.embedding is None:
                    continue

                # Cosine similarity to semantic phrase
                base_sim = float(np.dot(trk.embedding, text_vec))
                if aux_vecs:
                    aux_sim = max(float(np.dot(trk.embedding, av)) for av in aux_vecs)
                    sim = max(base_sim, aux_sim)
                else:
                    sim = base_sim

                # Color agreement adjustment
                color_bonus = 0.0
                if intent["detected_color"]:
                    if trk.dominant_color == intent["detected_color"]:
                        color_bonus = 0.08
                    elif intent["detected_color"] in trk.dominant_color:
                        color_bonus = 0.04

                # Category agreement adjustment
                cat_bonus = 0.0
                target_obj = intent["target_object"]
                if target_obj == "person":
                    if trk.label == "person":
                        cat_bonus = 0.25
                    else:
                        cat_bonus = -0.35  # Discourage vehicle tracks when querying for a person
                elif target_obj in {"vehicle", "car", "truck", "bus", "motorcycle"}:
                    if trk.label == "person":
                        cat_bonus = -0.35  # Discourage person tracks when querying for a vehicle
                    elif trk.label in {"car", "truck", "bus", "motorcycle"}:
                        cat_bonus = 0.05

                total_score = sim + color_bonus + cat_bonus

                # Check if query specifically asked about exit/left
                if intent["is_exit"]:
                    # Prioritize tracks that left the camera
                    is_exit_event = trk.left_camera or (trk.end_time > trk.start_time)
                    timestamp = trk.end_time
                    event_type = "left"
                    if trk.left_camera:
                        total_score += 0.05
                elif intent["is_entry"]:
                    timestamp = trk.start_time
                    event_type = "entered"
                else:
                    # Default: peak sighting or exit depending on query
                    timestamp = trk.best_time
                    event_type = "sighted"

                # Find associated journey or handover
                matching_journey = next(
                    (
                        j for j in journeys
                        if any(s["camera"] == cam_name and s["track_id"] == trk.track_id for s in j.sightings)
                    ),
                    None,
                )
                matching_ho = next(
                    (
                        h for h in handovers
                        if (h.from_camera == cam_name and h.from_track_id == trk.track_id)
                        or (h.to_camera == cam_name and h.to_track_id == trk.track_id)
                    ),
                    None,
                )

                candidates.append(
                    GroundedMatch(
                        camera_name=cam_name,
                        timestamp=timestamp,
                        timestamp_str=format_timestamp(timestamp),
                        crop=trk.best_crop,
                        confidence=total_score,
                        label=trk.label,
                        track_id=trk.track_id,
                        event_type=event_type,
                        dominant_color=trk.dominant_color,
                        details=(
                            f"Observed from {format_timestamp(trk.start_time)} to {format_timestamp(trk.end_time)}. "
                            f"{'Exited camera view.' if trk.left_camera else 'Remained in frame.'}"
                        ),
                        journey=matching_journey,
                        handover=matching_ho,
                    )
                )

        if not candidates:
            return ConversationalResult(
                query=clean_query,
                resolved_query=self.clarify_session.resolve_query(clean_query),
                answer_text=f"No matching visual events found for **{clean_query}** across the indexed cameras.",
                grounded_matches=[],
            )

        # Sort candidates by visual confidence score descending
        candidates.sort(key=lambda m: m.confidence, reverse=True)
        primary = candidates[0]

        # Build grounded natural language response
        timeline_steps: list[dict[str, Any]] = []

        icon = "🚶" if (primary.label == "person" or intent["target_object"] == "person") else "🚗"
        subj_name = f"person ({intent['semantic_phrase']})" if primary.label == "person" else f"{intent['semantic_phrase']}"

        if intent["is_exit"]:
            time_display = f"**{primary.timestamp:.2f}s** ({primary.timestamp_str})"
            ans_lines = [
                f"### {icon} Grounded Evidence: {intent['semantic_phrase'].title()} Departure",
                f"The **{subj_name}** left **{primary.camera_name}** at {time_display}.",
                "",
                f"• **Camera Source**: `{primary.camera_name}`",
                f"• **Departure Timestamp**: `{primary.timestamp:.2f}s` ({primary.timestamp_str})",
                f"• **Event Status**: `EXIT / LEFT CAMERA VIEW`",
                f"• **Visual Confidence**: `{max(0.0, primary.confidence):.1%}` (SigLIP zero-shot grounding)",
                f"• **Track ID**: #{primary.track_id} (`{primary.dominant_color} {primary.label}`)",
            ]

            # Cross-camera timeline reconstruction if available
            if primary.journey and primary.journey.has_handover:
                ans_lines.append("")
                ans_lines.append("#### 🌐 Cross-Camera Continuity & Handover Path")
                for idx, s in enumerate(primary.journey.sightings):
                    is_curr = s["camera"] == primary.camera_name
                    marker = "🚪 **Left**" if s.get("is_exit") else ("👀 **Entered**" if s.get("is_entry") else "📡 **Observed**")
                    ans_lines.append(
                        f"{idx+1}. {marker} `{s['camera']}` at **{format_timestamp(s['start_time'])}** – **{format_timestamp(s['end_time'])}**"
                        + (" *(Queried Departure Point)*" if is_curr else "")
                    )
                    timeline_steps.append({
                        "camera": s["camera"],
                        "start_time": s["start_time"],
                        "end_time": s["end_time"],
                        "crop": s["crop"],
                        "status": "exit" if s.get("is_exit") else "sighting",
                    })

                if primary.handover:
                    ans_lines.append(
                        f"\n⏱️ **Handover Transition:** Transited from `{primary.handover.from_camera}` to `{primary.handover.to_camera}` with a delay of **{primary.handover.delay_seconds:.1f}s** ({primary.handover.similarity:.1%} Re-ID match)."
                    )

        elif intent["is_entry"]:
            time_display = f"**{primary.timestamp:.2f}s** ({primary.timestamp_str})"
            ans_lines = [
                f"### {icon} Grounded Evidence: {intent['semantic_phrase'].title()} Arrival",
                f"The **{subj_name}** was first observed entering **{primary.camera_name}** at {time_display}.",
                "",
                f"• **Camera Source**: `{primary.camera_name}`",
                f"• **Arrival Timestamp**: `{primary.timestamp:.2f}s` ({primary.timestamp_str})",
                f"• **Visual Confidence**: `{max(0.0, primary.confidence):.1%}`",
                f"• **Track ID**: #{primary.track_id} (`{primary.dominant_color} {primary.label}`)",
            ]
        else:
            time_display = f"**{primary.timestamp:.2f}s** ({primary.timestamp_str})"
            ans_lines = [
                f"### {icon} Grounded Evidence Found",
                f"Found matching **{subj_name}** on **{primary.camera_name}** at {time_display}.",
                "",
                f"• **Camera Source**: `{primary.camera_name}`",
                f"• **Timestamp**: `{primary.timestamp:.2f}s` ({primary.timestamp_str})",
                f"• **Visual Match**: `{max(0.0, primary.confidence):.1%}`",
                f"• **Track ID**: #{primary.track_id} (`{primary.dominant_color} {primary.label}`)",
                f"• **Details**: {primary.details}",
            ]
            if primary.journey and primary.journey.has_handover:
                cams = " ➔ ".join(s["camera"] for s in primary.journey.sightings)
                ans_lines.append(f"• **Reconstructed Journey**: {cams}")

        answer_text = "\n".join(ans_lines)

        return ConversationalResult(
            query=clean_query,
            resolved_query=self.clarify_session.resolve_query(clean_query),
            answer_text=answer_text,
            grounded_matches=candidates[:6],
            primary_match=primary,
            reconstructed_timeline=timeline_steps,
            metadata={
                "intent": intent,
                "resolved_camera": resolved_camera,
                "total_candidates": len(candidates),
            },
        )
