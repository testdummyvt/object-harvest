from pathlib import Path
import json

from object_harvest.cli import main


def test_cli_smoke(tmp_path: Path, monkeypatch):
    # Prepare simple image using shared test image
    from pathlib import Path as _Path
    img_path = _Path("tests/data/test.jpeg")
    img = tmp_path / "a.png"
    img.write_bytes(img_path.read_bytes())

    # Monkeypatch pipeline.process_images to avoid network
    import object_harvest.pipeline as pipeline

    def fake_process(cfg):  # noqa: D401
        out = cfg.output
        out.write_text(json.dumps({"image_id": "a", "objects": [{"name": "cat"}]}) + "\n")

    monkeypatch.setattr(pipeline, "process_images", fake_process)

    out = tmp_path / "out.jsonl"
    argv = ["--source", str(tmp_path), "--model", "dummy", "--output", str(out)]
    rc = main(argv)
    assert rc == 0
    assert out.exists()
    content = out.read_text().strip()
    assert "cat" in content
