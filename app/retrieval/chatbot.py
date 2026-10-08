"""Free, local-first chatbot for multi-stream video intelligence.

Provides accurate, grounded conversational question-answering with virtual/visual
evidence (camera, exact timestamp, object crops, cross-camera timelines).

Supports:
1. Fast Local Grounded Agent: Deterministic, instant zero-latency responses grounded in SigLIP/YOLO detections.
2. GPT4All Local LLM: Free, open-source local GGUF models (e.g. Llama-3.2-1B-Instruct, Orca-mini) running 100% on-device.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.retrieval.conversational_engine import (
    ConversationalResult,
    GroundedMatch,
    MultiStreamConversationalEngine,
    format_timestamp,
)

logger = logging.getLogger(__name__)


@dataclass
class ChatbotMessage:
    """A conversational turn in the video intelligence chat."""

    role: str  # 'user' or 'assistant'
    content: str
    grounded_result: ConversationalResult | None = None
    primary_match: GroundedMatch | None = None
    evidence_crops: list[np.ndarray] = field(default_factory=list)
    timeline_steps: list[dict[str, Any]] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_entity: str | None = None
    clarification_prompt: str | None = None
    model_used: str = "Local Grounded Agent"


import os


class _suppress_c_stderr:
    """Context manager to suppress low-level C runtime stderr (e.g. harmless CUDA probe messages)."""

    def __enter__(self):
        try:
            self.null_fd = os.open(os.devnull, os.O_RDWR)
            self.old_stderr_fd = os.dup(2)
            os.dup2(self.null_fd, 2)
        except Exception:
            self.old_stderr_fd = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if getattr(self, "old_stderr_fd", None) is not None:
            try:
                os.dup2(self.old_stderr_fd, 2)
                os.close(self.old_stderr_fd)
                os.close(self.null_fd)
            except Exception:
                pass


class VideoIntelligenceChatbot:
    """Conversational multi-stream CCTV assistant."""

    def __init__(
        self,
        engine: MultiStreamConversationalEngine,
        gpt4all_model_name: str = "Llama-3.2-1B-Instruct-Q4_0.gguf",
    ) -> None:
        self.engine = engine
        self.gpt4all_model_name = gpt4all_model_name
        self._gpt4all_instance: Any = None
        self._gpt4all_available: bool = False
        self._check_gpt4all()

    def _check_gpt4all(self) -> bool:
        """Check if gpt4all library is importable."""
        try:
            import gpt4all
            self._gpt4all_available = True
            return True
        except ImportError:
            self._gpt4all_available = False
            return False

    @property
    def is_gpt4all_installed(self) -> bool:
        return self._gpt4all_available

    @property
    def is_gpt4all_loaded(self) -> bool:
        return self._gpt4all_instance is not None

    def load_gpt4all(self, model_name: str | None = None) -> bool:
        """Load local GPT4All model into memory cleanly without verbose C++ probe warnings."""
        if not self._gpt4all_available:
            return False
        import gpt4all

        target_model = model_name or self.gpt4all_model_name
        try:
            with _suppress_c_stderr():
                self._gpt4all_instance = gpt4all.GPT4All(
                    model_name=target_model,
                    device="cpu",
                    allow_download=True,
                )
            self.gpt4all_model_name = target_model
            return True
        except Exception as exc:
            logger.warning(f"Could not load GPT4All model '{target_model}': {exc}")
            self._gpt4all_instance = None
            return False


    def chat(
        self,
        user_query: str,
        multicam_results: dict[str, Any] | None,
        history: list[dict[str, Any]] | None = None,
        use_gpt4all: bool = False,
    ) -> ChatbotMessage:
        """Process a user query, retrieve grounded evidence, and generate an accurate response."""
        clean_query = user_query.strip()
        if not clean_query:
            return ChatbotMessage(
                role="assistant",
                content="Please ask a question about your CCTV video footage (e.g., 'when did the yellow car leave?').",
            )

        # 1. Retrieve grounded visual facts from the multi-camera intelligence engine
        conv_res: ConversationalResult = self.engine.answer_query(clean_query, multicam_results)

        # Handle Clarify-Once requirement
        if conv_res.needs_clarification:
            return ChatbotMessage(
                role="assistant",
                content=conv_res.answer_text,
                grounded_result=conv_res,
                needs_clarification=True,
                clarification_entity=conv_res.clarification_entity,
                clarification_prompt=conv_res.clarification_prompt,
                model_used="Clarify-Once Memory",
            )

        # Gather visual evidence
        primary = conv_res.primary_match
        evidence_crops = [m.crop for m in conv_res.grounded_matches if m.crop is not None]
        timeline_steps = conv_res.reconstructed_timeline

        # 2. Synthesize accurate conversational answer
        # If GPT4All is requested and loaded, generate grounded natural language synthesis with the LLM
        if use_gpt4all and self._gpt4all_instance is not None and primary is not None:
            try:
                # Prepare evidence summary for the LLM
                evidence_prompt = (
                    "You are BiggBoss AI, a multi-camera CCTV intelligence assistant. "
                    "Answer the user's question accurately using ONLY the verified evidence facts below. "
                    "Always mention the camera name, exact timestamp in seconds, and visual confirmation.\n\n"
                    f"VERIFIED EVIDENCE:\n"
                    f"- User Question: {clean_query}\n"
                    f"- Camera: {primary.camera_name}\n"
                    f"- Timestamp: {primary.timestamp:.2f} seconds ({primary.timestamp_str})\n"
                    f"- Event Type: {primary.event_type.upper()}\n"
                    f"- Vehicle/Object: {primary.dominant_color} {primary.label}\n"
                    f"- Visual Match Confidence: {primary.confidence:.1%}\n"
                    f"- Details: {primary.details}\n"
                )
                if timeline_steps:
                    cams = " -> ".join(s["camera"] for s in timeline_steps)
                    evidence_prompt += f"- Cross-Camera Path: {cams}\n"

                evidence_prompt += "\nAnswer the question in 2-3 clear, authoritative sentences with the verified camera and timestamp:"

                llm_reply = self._gpt4all_instance.generate(
                    prompt=evidence_prompt,
                    max_tokens=150,
                    temp=0.2,  # Low temperature for strict factual accuracy
                ).strip()

                final_content = (
                    f"{llm_reply}\n\n"
                    f"📌 **Grounded Verification:**\n"
                    f"• **Camera:** `{primary.camera_name}`\n"
                    f"• **Timestamp:** `{primary.timestamp:.2f}s` ({primary.timestamp_str})\n"
                    f"• **Event:** `{primary.event_type.upper()}`\n"
                    f"• **Visual Confidence:** `{primary.confidence:.1%}`"
                )
                model_name = f"GPT4All ({self.gpt4all_model_name})"
            except Exception as e:
                logger.warning(f"GPT4All generation failed, falling back to local grounded agent: {e}")
                final_content = conv_res.answer_text
                model_name = "Local Grounded Agent"
        else:
            final_content = conv_res.answer_text
            model_name = "Local Grounded Agent"

        return ChatbotMessage(
            role="assistant",
            content=final_content,
            grounded_result=conv_res,
            primary_match=primary,
            evidence_crops=evidence_crops,
            timeline_steps=timeline_steps,
            model_used=model_name,
        )
