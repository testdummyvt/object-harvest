from pathlib import Path

from object_harvest.pipeline import process_images
from object_harvest.schemas import RunConfig

# Mock VLM by monkeypatching VLMClient methods

class DummyClient:
    def __init__(self, *a, **kw):
        pass

    def list_objects(self, image_path):  # noqa: D401
        return ([type("O", (), {"name": "cat", "confidence": 0.9})()], None)

    def list_boxes(self, image_path, object_list, size=None):  # noqa: D401
        return ([type("B", (), {"name": "cat", "x1": 10, "y1": 15, "x2": 80, "y2": 90, "confidence": 0.8})()], None)


def test_pipeline_with_mock(tmp_path: Path, monkeypatch):
    from pathlib import Path as _Path
    img_path = _Path("tests/data/test.jpeg")
    img = tmp_path / "a.png"
    img.write_bytes(img_path.read_bytes())

    from object_harvest import pipeline

    monkeypatch.setattr(pipeline, "VLMClient", lambda *a, **k: DummyClient())
    out = tmp_path / "out.jsonl"
    cfg = RunConfig(
        source_dir=tmp_path,
        list_file=None,
        dataset=None,
        output=out,
        model="dummy",
        boxes=True,
        threads=2,
        api_base=None,
        api_key=None,
    )
    process_images(cfg)
    data = out.read_text().strip().splitlines()
    assert len(data) == 1
    assert "cat" in data[0]
