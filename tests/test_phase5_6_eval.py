from app.eval.ablation import ablation_report
from app.hardening.status import restart_status
from app.ui.clarify_flow import ClarifySession


def test_ablation_report_shows_baseline_vs_variants():
    report = ablation_report()

    assert report["baseline"] == 0.82
    assert report["variants"]["B1"]["wins_baseline"] is False
    assert report["variants"]["B2"]["wins_baseline"] is False
    assert report["variants"]["B3"]["wins_baseline"] is True
    assert report["variants"]["B4"]["wins_baseline"] is True
    assert report["variants"]["B5"]["wins_baseline"] is True


def test_restart_status_and_clarify_restart_flow():
    session = ClarifySession()
    assert restart_status(session.is_running)["status"] == "healthy"

    session.kill()
    assert restart_status(session.is_running)["status"] == "stopped"

    session.restart()
    assert restart_status(session.is_running)["status"] == "healthy"
