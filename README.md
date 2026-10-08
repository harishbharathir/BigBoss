# Multi-Stream Video Intelligence

Local-first video indexing and natural-language retrieval for multi-camera footage.
The project is being delivered in validated phases; only the completed phase is
considered supported. All runtime configuration is in `config.yaml`.

## Phase 0 — setup and model/GPU sanity check

Requirements: Windows, Python 3.11, NVIDIA GPU with a recent driver (optional;
CPU mode works), FFmpeg on `PATH` for later clip generation, and Git.

1. Create and activate an isolated Python 3.11 environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

2. Install PyTorch using the current command from the official selector at
   <https://pytorch.org/get-started/locally/>. For CPU-only testing, install the
   CPU build there. Then install the application dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Cache the configured detector and SigLIP weights, then load both together
   and report peak CUDA memory:

   ```powershell
   python scripts/download_models.py
   python scripts/gpu_sanity.py
   ```

Weights are cached under `data/models` and can be loaded locally after the
initial download. Model downloads require network access once. The model check
fails if peak allocated CUDA memory exceeds `runtime.max_vram_gb`.

## Development status

- **Multi-Camera Vehicle Tracker (Cross-Camera Re-ID)**: Ingests 2+ CCTV footage files
  (such as `gate_cam.mp4` and `rear_cam.mp4`), tracks vehicles within each camera,
  extracts deep visual appearance embeddings using SigLIP, associates matching vehicle
  identities across camera views, and detects transitions (e.g. car left Gate Cam ➔
  appeared on Rear Cam) with side-by-side visual crop comparisons and handover delay metrics.
- **Single-Stream Semantic Search**: Local MP4 selection/upload, sampled-frame YOLO-World
  detection, free-form SigLIP image-text ranking, annotated frames, and CSV export.
- Prototype modules also cover metadata inspection, lightweight tracking,
  spatial-memory helpers, clarify-session state, alerts, and RTSP stream opening.

### Run the dashboard

From the project root, activate `.venv`, install `requirements.txt`, and run:

```powershell
streamlit run app/ui/dashboard.py
```

In the sidebar, switch between:
1. **🚗 Multi-Camera Vehicle Tracker (Re-ID)**: Select multiple CCTV feeds (or upload clips),
   configure sampling and similarity thresholds, and click **Track Vehicles Across Cameras**.
   Inspect chronological handover cards, visual similarity match percentages, transition delays,
   and download CSV handover reports.
2. **🔍 Single-Stream Natural Search**: Query a single video using natural-language phrases
   (e.g., *a person in a red shirt*, *a vehicle turning*).