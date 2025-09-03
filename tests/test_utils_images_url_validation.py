import pytest
from object_harvest.utils.images import load_image_from_item, image_part_from_item


def test_load_image_from_item_rejects_file_scheme(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"not-an-image")
    with pytest.raises(ValueError):
        load_image_from_item({"url": f"file://{p}"})


def test_image_part_from_item_rejects_file_scheme(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        image_part_from_item({"url": f"file://{p}"})


def test_image_part_from_item_accepts_http():
    url = "http://example.com/a.jpg"
    part = image_part_from_item({"url": url})
    assert part["image_url"]["url"] == url


def test_image_part_from_item_accepts_https():
    url = "https://example.com/a.jpg"
    part = image_part_from_item({"url": url})
    assert part["image_url"]["url"] == url
