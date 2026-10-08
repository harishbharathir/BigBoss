"""Local-first Streamlit dashboard for video search and multi-camera vehicle tracking."""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path when running via `streamlit run app/ui/dashboard.py`
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from app.config import load_config
from app.reid.cross_camera import analyze_multiple_cameras
from app.retrieval.visual_search import analyze_video, load_visual_models, search_frames
from app.ui.clarify_flow import ClarifySession


@st.cache_resource(show_spinner=False)
def get_visual_models() -> tuple[Any, Any, Any, str]:
    return load_visual_models()


@st.cache_data(show_spinner="Extracting YouTube stream...", ttl=900)
def extract_youtube_stream_url(url: str) -> str:
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("Please install yt-dlp to support YouTube live streams: pip install yt-dlp")

    runtimes = {rt: {} for rt in ["node", "deno", "quickjs"] if shutil.which(rt)}
    ydl_opts: dict[str, Any] = {
        "format": "bestvideo[ext=mp4]/best[ext=mp4]/bestvideo/best",
        "quiet": True,
        "no_warnings": True,
    }
    if runtimes:
        ydl_opts["js_runtimes"] = runtimes

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        stream_url = info.get("url")
        if not stream_url and "entries" in info and info["entries"]:
            stream_url = info["entries"][0].get("url")
        if not stream_url:
            raise ValueError("No direct stream URL could be found for this YouTube video.")
        return stream_url


st.set_page_config(
    page_title="BiggBoss | Multi-Stream Video Intelligence",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(ellipse at 15% 0%, #15263d 0, #090e17 45%, #05080e 100%); color: #e2e8f0; }
    [data-testid="stSidebar"] { background: #0b121d; border-right: 1px solid #1f2d40; }
    .hero {
        padding: 1.5rem 1.8rem; border: 1px solid #23344a; border-radius: 18px;
        background: linear-gradient(135deg, rgba(21,38,61,.94), rgba(12,20,32,.92));
        margin-bottom: 1.3rem; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .hero h1 { margin: 0; letter-spacing: -.03em; font-size: 2.1rem; color: #f8fafc; }
    .hero p { color: #9bb0c7; margin: .55rem 0 0; font-size: 1.02rem; }
    .badge {
        display: inline-block; padding: .28rem .75rem; border-radius: 999px;
        background: #113630; color: #5eead4; border: 1px solid #1f685c;
        font-size: .8rem; font-weight: 600; letter-spacing: .03em; margin-bottom: .6rem;
    }
    .badge-accent {
        display: inline-block; padding: .28rem .75rem; border-radius: 999px;
        background: #1e293b; color: #60a5fa; border: 1px solid #3b82f6;
        font-size: .8rem; font-weight: 600; letter-spacing: .03em; margin-bottom: .6rem;
    }
    [data-testid="stMetric"] {
        background: rgba(16,26,40,.85); border: 1px solid #22344c;
        padding: 1.1rem; border-radius: 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .handover-card {
        background: rgba(14,23,37,.92); border: 1px solid #253952;
        border-radius: 18px; padding: 1.3rem; margin-bottom: 1.3rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35); transition: border-color 0.2s ease;
    }
    .handover-card:hover {
        border-color: #3b82f6;
    }
    .handover-banner {
        background: linear-gradient(90deg, rgba(30,58,95,0.7), rgba(16,36,65,0.7));
        border: 1px solid #2b4970; border-radius: 12px;
        padding: 0.75rem 1rem; margin: 0.8rem 0 1.1rem 0;
        display: flex; align-items: center; justify-content: space-between;
        font-weight: 500; font-size: 0.96rem; color: #e2e8f0;
    }
    .badge-match {
        background: #0f3f38; color: #4ade80; border: 1px solid #1b6357;
        border-radius: 999px; padding: 0.25rem 0.8rem; font-size: 0.84rem; font-weight: 600;
    }
    .badge-cam {
        background: #1e293b; color: #93c5fd; border: 1px solid #334155;
        border-radius: 8px; padding: 0.2rem 0.55rem; font-size: 0.82rem; font-weight: 500;
    }
    .transition-flow {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        height: 100%; min-height: 140px; text-align: center;
        padding: 0.5rem;
    }
    .result-card {
        background: rgba(15,24,38,.88); border: 1px solid #24364c;
        border-radius: 16px; padding: .9rem; margin-bottom: 1rem;
    }
    .muted { color: #94a3b8; }
    </style>
    """,
    unsafe_allow_html=True,
)

config = load_config()
project_root: Path = config["_project_root"]
video_dir: Path = config["_paths"]["videos"]
video_dir.mkdir(parents=True, exist_ok=True)

with st.sidebar:
    st.markdown("### 🎛️ Navigation & Mode")
    app_mode = st.radio(
        "Select Workflow",
        [
            "🚗 Multi-Camera Vehicle Tracker (Re-ID)",
            "🔍 Single-Stream Natural Search",
        ],
        index=0,
    )
    st.divider()

# =========================================================================
# MODE 1: MULTI-CAMERA VEHICLE TRACKING & RE-ID
# =========================================================================
if app_mode == "🚗 Multi-Camera Vehicle Tracker (Re-ID)":
    st.markdown(
        """
        <div class="hero">
          <span class="badge">CROSS-CAMERA VEHICLE TRACKING · SIGLIP Re-ID + YOLO-WORLD</span>
          <h1>Multi-Camera CCTV Vehicle Tracker</h1>
          <p>Track vehicles across multiple camera streams, detect cross-camera handovers (vehicle left Camera A ➔ appeared on Camera B), and inspect matching visual evidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 📹 CCTV Cameras Selection")
        source_type = st.radio(
            "Camera Sources",
            ["Project CCTV Videos (Gate Cam & Rear Cam)", "Upload Multiple CCTV Clips"],
            index=0,
        )

        camera_sources_to_run: list[dict[str, Any]] = []

        if source_type == "Project CCTV Videos (Gate Cam & Rear Cam)":
            available_videos = sorted(
                path for path in video_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
            )
            if not available_videos:
                st.warning("No video files found in data/videos.")
            else:
                default_selected = [
                    v for v in available_videos if v.name in {"gate_cam.mp4", "rear_cam.mp4"}
                ]
                if not default_selected:
                    default_selected = available_videos[:2]

                selected_clips = st.multiselect(
                    "Choose CCTV footage files",
                    available_videos,
                    default=default_selected,
                    format_func=lambda p: p.name,
                )

                st.markdown("#### Camera Configurations")
                for i, clip in enumerate(selected_clips):
                    default_name = "Gate Cam" if "gate" in clip.name.lower() else ("Rear Cam" if "rear" in clip.name.lower() else f"Camera {i+1}")
                    col_a, col_b = st.columns([3, 2])
                    cam_name = col_a.text_input(f"Name #{i+1}", value=default_name, key=f"name_{clip.name}")
                    cam_offset = col_b.number_input(f"Offset (s) #{i+1}", value=0.0, step=0.5, key=f"offset_{clip.name}")
                    camera_sources_to_run.append({
                        "name": cam_name,
                        "source": str(clip.resolve()),
                        "time_offset": cam_offset,
                    })

        else:
            uploaded_files = st.file_uploader(
                "Upload 2 or more CCTV video clips",
                type=["mp4", "mov", "mkv", "avi"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                st.markdown("#### Uploaded Cameras")
                for i, file_obj in enumerate(uploaded_files):
                    payload = file_obj.getvalue()
                    digest = hashlib.sha256(payload).hexdigest()[:12]
                    suffix = Path(file_obj.name).suffix.lower() or ".mp4"
                    temp_path = Path(tempfile.gettempdir()) / f"biggboss-cctv-{digest}{suffix}"
                    if not temp_path.exists() or temp_path.stat().st_size != len(payload):
                        temp_path.write_bytes(payload)
                    col_a, col_b = st.columns([3, 2])
                    cam_name = col_a.text_input(f"Cam #{i+1} Name", value=f"Camera {i+1} ({Path(file_obj.name).stem})", key=f"up_name_{digest}")
                    cam_offset = col_b.number_input(f"Offset (s)", value=0.0, step=0.5, key=f"up_off_{digest}")
                    camera_sources_to_run.append({
                        "name": cam_name,
                        "source": str(temp_path),
                        "time_offset": cam_offset,
                    })

        st.markdown("### ⚙️ Re-ID Parameters")
        sample_fps = st.slider("Analysis Sampling Rate (FPS)", min_value=1.0, max_value=5.0, value=2.5, step=0.5)
        sim_threshold = st.slider("Re-ID Similarity Threshold", min_value=0.50, max_value=0.95, value=0.70, step=0.05)
        max_handover_sec = st.slider("Max Handover Window (seconds)", min_value=5.0, max_value=90.0, value=45.0, step=5.0)
        conf_threshold = st.slider("Min Vehicle Detection Conf", min_value=0.20, max_value=0.70, value=0.35, step=0.05)

        run_multicam = st.button("⚡ Track Vehicles Across Cameras", type="primary", use_container_width=True)
        st.divider()
        st.caption("Deep SigLIP vision embeddings match vehicle identity across views.")

    # Execution trigger
    if run_multicam:
        if len(camera_sources_to_run) < 2:
            st.warning("Please configure at least 2 camera feeds to track vehicles across cameras.")
        else:
            try:
                progress_bar = st.progress(0, text="Initializing local detector and vision embedder...")
                models = get_visual_models()

                def update_progress(ratio: float, msg: str) -> None:
                    progress_bar.progress(min(1.0, max(0.0, ratio)), text=msg)

                results = analyze_multiple_cameras(
                    camera_sources=camera_sources_to_run,
                    detector=models[0],
                    processor=models[1],
                    embedder=models[2],
                    device=models[3],
                    sample_fps=sample_fps,
                    similarity_threshold=sim_threshold,
                    max_handover_seconds=max_handover_sec,
                    conf_threshold=conf_threshold,
                    progress_callback=update_progress,
                )
                st.session_state["multicam_results"] = results
                st.session_state["multicam_config"] = {
                    "cams": [c["name"] for c in camera_sources_to_run],
                    "sim_threshold": sim_threshold,
                }
                progress_bar.empty()
                st.success(
                    f"Successfully analyzed {len(results['cameras'])} cameras! "
                    f"Identified {len(results['handovers'])} cross-camera handovers across {results['total_vehicles_count']} unique vehicles."
                )
            except Exception as exc:
                st.error(f"Multi-camera analysis failed: {exc}")

    multicam_results = st.session_state.get("multicam_results")

    if multicam_results is None:
        st.markdown("### 📋 How Multi-Camera Tracking Works")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                #### 1. Ingestion & Detection
                Each CCTV stream is sampled and processed using **YOLO-World**. Vehicle bounding boxes (`car`, `truck`, `bus`, `motorcycle`) are detected with sub-second accuracy.
                """
            )
        with col2:
            st.markdown(
                """
                #### 2. Deep Visual Re-ID
                Detected vehicle crops are embedded using **SigLIP** (768-dimensional normalized visual vectors) along with dominant color extraction, creating robust appearance signatures.
                """
            )
        with col3:
            st.markdown(
                """
                #### 3. Cross-Camera Handover
                The Re-ID engine links tracklets across cameras:
                - Detects when a car exits Camera A
                - Tracks handover transition delay $\Delta t$
                - Identifies reappearance on Camera B with similarity verification.
                """
            )

        # st.info("💡 **Ready to run:** Select **Gate Cam** and **Rear Cam** in the sidebar and click **'⚡ Track Vehicles Across Cameras'** to see vehicles tracked across both cameras.")

    else:
        # Display Metrics
        handovers = multicam_results["handovers"]
        journeys = multicam_results["journeys"]
        cameras = multicam_results["cameras"]

        metric_cols = st.columns(4)
        metric_cols[0].metric("Cameras Ingested", str(len(cameras)))
        metric_cols[1].metric("Cross-Camera Handovers", str(len(handovers)))
        metric_cols[2].metric("Total Unique Vehicles", str(multicam_results["total_vehicles_count"]))
        avg_delay = multicam_results.get("average_handover_delay", 0.0)
        metric_cols[3].metric("Avg Handover Delay", f"{avg_delay:.1f} s")

        st.divider()

        # Filtering and search options
        filter_col1, filter_col2 = st.columns([3, 2])
        view_filter = filter_col1.radio(
            "Filter View",
            ["🚗➡️ Cross-Camera Handovers Only", "🌐 All Tracked Vehicles", "⏱️ Sequential Only (Left ➔ Appeared)"],
            horizontal=True,
        )
        search_kw = filter_col2.text_input("Filter by color/label", placeholder="e.g. silver, red, car")

        # Handover events section
        filtered_handovers = handovers
        if view_filter == "⏱️ Sequential Only (Left ➔ Appeared)":
            filtered_handovers = [h for h in handovers if h.delay_seconds >= 0]

        if search_kw.strip():
            kw = search_kw.strip().lower()
            filtered_handovers = [
                h for h in filtered_handovers
                if kw in h.label.lower() or kw in h.dominant_color.lower() or kw in h.description.lower()
            ]

        st.markdown(f"### 🎯 Detected Vehicle Handovers ({len(filtered_handovers)} Events)")

        if not filtered_handovers and view_filter != "🌐 All Tracked Vehicles":
            st.info("No handovers match the current filter or similarity threshold.")

        for ho in filtered_handovers:
            with st.container():
                st.markdown('<div class="handover-card">', unsafe_allow_html=True)

                # Header with Vehicle ID and Match Badge
                header_col1, header_col2 = st.columns([3, 1])
                color_name = ho.dominant_color.capitalize() if ho.dominant_color != "unknown" else ""
                header_col1.markdown(
                    f"### 🚗 Vehicle #{ho.global_id} · <span style='color:#60a5fa;'>{color_name} {ho.label.capitalize()}</span>",
                    unsafe_allow_html=True,
                )
                header_col2.markdown(
                    f"<div style='text-align:right;'><span class='badge-match'>{ho.similarity:.1%} Visual Match</span></div>",
                    unsafe_allow_html=True,
                )

                # Handover Flow Banner
                if ho.delay_seconds >= 0:
                    delay_text = f"⏱️ Transition: {ho.delay_seconds:.1f}s"
                    flow_html = f"""
                    <div class="handover-banner">
                      <span>🚪 <b>Left {ho.from_camera}</b> at {ho.exit_time:.1f}s</span>
                      <span style="color:#60a5fa; font-weight:600;">➔ {delay_text} ➔</span>
                      <span>👀 <b>Appeared on {ho.to_camera}</b> at {ho.entry_time:.1f}s</span>
                    </div>
                    """
                else:
                    flow_html = f"""
                    <div class="handover-banner">
                      <span>📡 <b>Co-observed on {ho.from_camera}</b> (exit {ho.exit_time:.1f}s)</span>
                      <span style="color:#38bdf8; font-weight:600;">⇄ Multi-Angle Sighting ⇄</span>
                      <span>📡 <b>Observed on {ho.to_camera}</b> (entry {ho.entry_time:.1f}s)</span>
                    </div>
                    """
                st.markdown(flow_html, unsafe_allow_html=True)

                # Side-by-side Visual Crops Comparison
                c1, c2, c3 = st.columns([4, 2, 4])
                with c1:
                    st.markdown(f"**From: {ho.from_camera}**")
                    st.image(ho.from_crop, caption=f"Last seen at {ho.exit_time:.1f}s", use_container_width=True)

                with c2:
                    st.markdown('<div class="transition-flow">', unsafe_allow_html=True)
                    st.markdown(f"**Re-ID Match**")
                    st.markdown(f"<span style='font-size:1.4rem; color:#4ade80;'><b>{ho.similarity:.1%}</b></span>", unsafe_allow_html=True)
                    if ho.delay_seconds >= 0:
                        st.markdown(f"<span class='badge-cam'>Delay: {ho.delay_seconds:.1f}s</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='badge-cam'>Concurrent</span>", unsafe_allow_html=True)
                    st.markdown(f"<span class='muted' style='font-size:0.8rem; margin-top:0.4rem;'>Color: {ho.dominant_color}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with c3:
                    st.markdown(f"**To: {ho.to_camera}**")
                    st.image(ho.to_crop, caption=f"First seen at {ho.entry_time:.1f}s", use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)

        # Full Journey Explorer
        if view_filter == "🌐 All Tracked Vehicles":
            st.divider()
            st.markdown(f"### 🌐 All Tracked Vehicles ({len(journeys)} Total)")
            for j in journeys:
                with st.expander(f"{j.title} · ({len(j.sightings)} Camera Sightings)", expanded=j.has_handover):
                    st.markdown(f"**Vehicle Class:** {j.label.capitalize()} | **Color:** {j.dominant_color.capitalize()}")
                    sighting_cols = st.columns(len(j.sightings))
                    for idx, s in enumerate(j.sightings):
                        with sighting_cols[idx]:
                            st.markdown(f"**{s['camera']}**")
                            st.image(s["crop"], caption=f"{s['start_time']:.1f}s – {s['end_time']:.1f}s", use_container_width=True)
                            if s["is_exit"]:
                                st.caption("🚪 Exited camera view")
                            if s["is_entry"]:
                                st.caption("👀 Entered camera view")

        # CSV Download Section
        st.divider()
        csv_rows = []
        for ho in handovers:
            csv_rows.append({
                "global_vehicle_id": ho.global_id,
                "label": ho.label,
                "dominant_color": ho.dominant_color,
                "from_camera": ho.from_camera,
                "exit_time_seconds": f"{ho.exit_time:.2f}",
                "to_camera": ho.to_camera,
                "entry_time_seconds": f"{ho.entry_time:.2f}",
                "transition_delay_seconds": f"{ho.delay_seconds:.2f}",
                "reid_visual_similarity": f"{ho.similarity:.4f}",
                "handover_type": ho.handover_type,
                "description": ho.description,
            })

        if csv_rows:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
            st.download_button(
                "📥 Download Cross-Camera Handover Report (CSV)",
                data=buffer.getvalue(),
                file_name="biggboss-cross-camera-handovers.csv",
                mime="text/csv",
            )


# =========================================================================
# MODE 2: SINGLE-STREAM SEMANTIC SEARCH
# =========================================================================
else:
    st.markdown(
        """
        <div class="hero">
          <span class="badge-accent">SINGLE-STREAM SEARCH · SIGLIP + YOLO-WORLD</span>
          <h1>Single-Stream Video Intelligence</h1>
          <p>Search sampled moments in your footage with a free-form natural-language query.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 🎛️ Single Stream Controls")
        source_mode = st.radio("Video source", ["Project video", "Upload MP4", "Live RTSP Stream"], horizontal=False)
        selected_video: Path | str | None = None
        source_key: str | None = None
        if source_mode == "Project video":
            local_videos = sorted(
                path for path in video_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}
            )
            if local_videos:
                selected_video = st.selectbox(
                    "Choose a clip", local_videos,
                    format_func=lambda path: path.name,
                )
                stat = selected_video.stat()
                source_key = f"{selected_video.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
            else:
                st.info("Add a video under data/videos or choose Upload MP4.")
        elif source_mode == "Upload MP4":
            uploaded = st.file_uploader("Choose a video", type=["mp4", "mov", "mkv", "avi"])
            if uploaded is not None:
                payload = uploaded.getvalue()
                digest = hashlib.sha256(payload).hexdigest()[:16]
                suffix = Path(uploaded.name).suffix.lower() or ".mp4"
                selected_video_path = Path(tempfile.gettempdir()) / f"biggboss-{digest}{suffix}"
                if not selected_video_path.exists() or selected_video_path.stat().st_size != len(payload):
                    selected_video_path.write_bytes(payload)
                selected_video = selected_video_path
                source_key = f"upload:{digest}"
        else:
            rtsp_url = st.text_input("RTSP / YouTube Live URL", placeholder="rtsp://... or https://youtube.com/...")
            if rtsp_url and rtsp_url.strip():
                raw_url = rtsp_url.strip()
                if "youtube.com" in raw_url or "youtu.be" in raw_url:
                    try:
                        selected_video = extract_youtube_stream_url(raw_url)
                    except Exception as e:
                        st.error(f"Failed to extract YouTube stream: {e}")
                        selected_video = None
                else:
                    selected_video = raw_url

                if selected_video:
                    source_key = f"stream:{hashlib.md5(raw_url.encode()).hexdigest()[:8]}"

        max_samples = st.slider("Frames to sample", min_value=4, max_value=120, value=32, step=4)
        top_k = st.slider("Results to show", min_value=2, max_value=12, value=6)
        analyze_clicked = st.button("⚡ Analyze video", type="primary", use_container_width=True)
        st.divider()
        st.caption("Configured classes: " + ", ".join(config["ingest"]["generic_vocabulary"]))

    if analyze_clicked:
        if selected_video is None or source_key is None:
            st.warning("Choose a local clip or upload a video first.")
        else:
            try:
                with st.spinner("Loading local models and sampling the video…"):
                    models = get_visual_models()
                    analysis = analyze_video(
                        selected_video,
                        max_samples=max_samples,
                        detector=models[0],
                        processor=models[1],
                        embedder=models[2],
                        device=models[3],
                    )
                st.session_state["visual_analysis"] = analysis
                st.session_state["analysis_key"] = (source_key, max_samples)
                st.session_state["visual_models"] = models
                st.success(f"Analyzed {analysis['sample_count']} frames.")
            except Exception as exc:
                st.error(f"Could not analyze this video: {exc}")

    analysis: dict[str, Any] | None = st.session_state.get("visual_analysis")
    analysis_key = st.session_state.get("analysis_key")
    if analysis is not None and (analysis_key is None or analysis_key != (source_key, max_samples)):
        analysis = None
        st.info("The selected video or sampling setting changed. Select **Analyze video** to refresh the index.")

    if analysis is None:
        st.markdown("## Start with a video")
        st.markdown(
            "Choose the sample clip or upload an MP4 in the sidebar, then analyze it. "
            "After indexing, search with phrases such as **a person carrying a bag**, "
            "**a red vehicle**, or **people gathered together**."
        )
    else:
        detections = sum(len(frame["boxes"]) for frame in analysis["frames"])
        columns = st.columns(4)
        columns[0].metric("Clip duration", f"{analysis['duration_seconds']:.1f} s")
        columns[1].metric("Frames indexed", str(analysis["sample_count"]))
        columns[2].metric("Object detections", str(detections))
        columns[3].metric("Sampling rate", f"{analysis['sample_count'] / max(analysis['duration_seconds'], 1):.2f} fps")

        st.markdown("### Ask a question about the footage")
        with st.form("natural_language_search", clear_on_submit=False):
            query_cols = st.columns([5, 1])
            query = query_cols[0].text_input(
                "Describe the moment or visual content",
                placeholder="e.g. a person in a red shirt near a vehicle",
                label_visibility="collapsed",
            )
            submitted = query_cols[1].form_submit_button("Search", type="primary", use_container_width=True)

        if submitted:
            st.session_state["last_query"] = query.strip()
        active_query = st.session_state.get("last_query", "")

        if active_query:
            if "clarify_session" not in st.session_state:
                st.session_state["clarify_session"] = ClarifySession()
            session = st.session_state["clarify_session"]
            clarification = session.clarify_once(active_query)

            if clarification:
                st.warning(clarification)
                with st.form("clarify_form"):
                    answer = st.text_input("Your answer")
                    if st.form_submit_button("Submit"):
                        session.learn(active_query, answer)
                        st.rerun()
            else:
                resolved_query = session.resolve_query(active_query)
                st.info(f"Resolved query: {resolved_query}")

                models = st.session_state.get("visual_models")
                if models is None:
                    try:
                        models = get_visual_models()
                        st.session_state["visual_models"] = models
                    except Exception as exc:
                        st.error(f"Could not load the local search models: {exc}")
                        models = None
                if models is not None:
                    try:
                        matches = search_frames(analysis, resolved_query, models[1], models[2], models[3])[:top_k]
                        st.markdown(f"#### Results for “{resolved_query}”")
                        st.caption("Ranked by SigLIP image–text similarity.")
                        result_cols = st.columns(2)
                        for index, match in enumerate(matches):
                            with result_cols[index % 2]:
                                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                                st.image(match["image"], use_container_width=True)
                                score = match["score"]
                                st.markdown(f"**{match['timestamp']:.2f}s** · similarity `{score:.3f}`")
                                labels = ", ".join(
                                    f"{box['label']} {box['confidence']:.0%}"
                                    for box in match["boxes"][:5]
                                ) or "No configured-class detections"
                                st.caption(f"Detected: {labels}")
                                st.markdown("</div>", unsafe_allow_html=True)

                        csv_rows = [
                            {
                                "rank": rank,
                                "timestamp_seconds": f"{match['timestamp']:.3f}",
                                "similarity": f"{match['score']:.6f}",
                                "detections": "; ".join(box["label"] for box in match["boxes"]),
                                "query": active_query,
                                "video": analysis["video"],
                            }
                            for rank, match in enumerate(matches, start=1)
                        ]
                        buffer = io.StringIO()
                        writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(csv_rows)
                        st.download_button(
                            "Download results as CSV",
                            data=buffer.getvalue(),
                            file_name="biggboss-search-results.csv",
                            mime="text/csv",
                        )
                    except Exception as exc:
                        st.error(f"Search failed: {exc}")
        else:
            st.info("Enter any natural-language query above to rank the indexed frames.")
