"""Fixtures for the BOX-CAPTURE tests: a simulated ESP32 behind a real pty."""
import pytest

from yq.box import settings as S


@pytest.fixture
def simbox():
    """simbox(time_scale=20, **BoxSimulator kwargs) -> connected Box driving the simulator."""
    from yq.box.device import Box
    from yq.box.sim import BoxSimulator
    from yq.box.simlink import PtyLink
    boxes = []

    def make(time_scale=20.0, heartbeat=True, **kw):
        link = PtyLink(BoxSimulator(time_scale=time_scale, **kw)).start()
        box = Box(port=link.port, heartbeat=heartbeat)
        box.sim, box.link = link.sim, link
        boxes.append(box)
        return box.connect(timeout=5)

    yield make
    for b in boxes:
        b.close()


@pytest.fixture
def mock_all(monkeypatch):
    """Whole box in mock mode (simulator + synthetic cameras/thermal), fast and small."""
    from yq.box import device
    from yq.common import config
    monkeypatch.setattr(config, "MOCK", True)
    monkeypatch.setattr(S, "SIM_TIME_SCALE", 50.0)
    monkeypatch.setattr(S, "MOCK_RESOLUTION", (320, 240))
    monkeypatch.setattr(device, "_box", None)
    yield
    if device._box is not None:
        device._box.close()
        device._box = None
