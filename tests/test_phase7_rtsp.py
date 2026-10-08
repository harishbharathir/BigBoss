from pathlib import Path

from app.rtsp.client import RTSPStream


def test_rtsp_stream_can_read_local_sample_video():
    sample = Path("data/videos/sample.mp4")
    stream = RTSPStream(sample)

    assert stream.open() is True
    ok, frame = stream.read_frame()
    assert ok is True
    assert frame is not None
    assert len(frame.shape) == 3
    stream.close()


def test_rtsp_stream_handles_invalid_url_without_crashing():
    stream = RTSPStream("rtsp://example.invalid/stream")

    assert stream.open() is False
    ok, frame = stream.read_frame()
    assert ok is False
    assert frame is None
