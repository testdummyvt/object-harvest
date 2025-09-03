from pathlib import Path
import base64
from PIL import Image

from object_harvest.utils.images import (
    load_image_bytes_jpeg,
    load_image_from_item,
    image_part_from_item,
)


def _make_sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    im = Image.new("RGB", (10, 6), color=(10, 20, 30))
    im.save(path)
    return path


def test_load_image_bytes_jpeg(tmp_path: Path):
    p = _make_sample_image(tmp_path)
    data = load_image_bytes_jpeg(str(p))
    assert isinstance(data, (bytes, bytearray))
    # Should be JPEG header
    assert data[:2] == b"\xff\xd8"


def test_load_image_from_item_path(tmp_path: Path):
    p = _make_sample_image(tmp_path)
    im = load_image_from_item({"path": str(p)})
    assert im.size == (10, 6)


def test_image_part_from_item_path(tmp_path: Path):
    p = _make_sample_image(tmp_path)
    part = image_part_from_item({"path": str(p)})
    assert part["type"] == "image_url"
    url = part["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    # Quick decode to verify valid base64
    b64 = url.split(",", 1)[1]
    data = base64.b64decode(b64)
    assert data[:2] == b"\xff\xd8"


def test_image_part_from_item_url():
    url = "https://example.com/image.jpg"
    part = image_part_from_item({"url": url})
    assert part["image_url"]["url"] == url
