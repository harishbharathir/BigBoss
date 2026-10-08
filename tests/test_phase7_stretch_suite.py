from app.alerts.rules import AlertManager
from app.privacy.redaction import PrivacyFilter
from app.rtsp.client import RTSPStream


def test_alert_manager_triggers_on_threshold():
    manager = AlertManager(threshold=0.8)
    result = manager.evaluate({"event": "person_detected", "score": 0.87})
    assert result["triggered"] is True
    assert result["message"] == "Alert triggered"


def test_privacy_filter_redacts_without_crashing():
    filter_obj = PrivacyFilter()
    assert filter_obj.redact(None) is None
    assert filter_obj.label(42) == "masked-42"


def test_rtsp_stream_can_open_demo_video():
    stream = RTSPStream("data/videos/sample.mp4")
    assert stream.open() is True
    ok, frame = stream.read_frame()
    assert ok is True
    assert frame is not None
    stream.close()
