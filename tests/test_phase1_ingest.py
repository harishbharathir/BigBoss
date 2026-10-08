from pathlib import Path

import pytest

from app.ingest import extract_video_metadata
from app.index import build_index_manifest


SAMPLE_VIDEO = Path("data/videos/sample.mp4")


def test_extract_video_metadata_for_sample_video():
    if not SAMPLE_VIDEO.exists():
        pytest.skip("sample video not present yet")

    metadata = extract_video_metadata(SAMPLE_VIDEO)

    assert metadata["width"] == 3840
    assert metadata["height"] == 2160
    assert metadata["duration_seconds"] > 20
    assert metadata["fps"] > 0


def test_build_index_manifest_contains_valid_fields():
    if not SAMPLE_VIDEO.exists():
        pytest.skip("sample video not present yet")

    manifest = build_index_manifest(SAMPLE_VIDEO)

    assert manifest["source_video"].endswith("sample.mp4")
    assert manifest["width"] == 3840
    assert manifest["height"] == 2160
    assert manifest["duration_seconds"] > 20
    assert manifest["index_status"] == "validated"
