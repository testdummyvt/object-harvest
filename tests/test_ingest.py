from pathlib import Path

from object_harvest.ingest import iter_folder, iter_list_file


def test_iter_folder(tmp_path: Path):
    # Use the shared test image for both cases
    img_path = Path("tests/data/test.jpeg")
    (tmp_path / "a.jpg").write_bytes(img_path.read_bytes())
    (tmp_path / "b.png").write_bytes(img_path.read_bytes())
    (tmp_path / "c.txt").write_text("nope")
    items = list(iter_folder(tmp_path))
    names = sorted(n for n, _ in items)
    assert names == ["a", "b"]


def test_iter_list_file(tmp_path: Path):
    img_path = Path("tests/data/test.jpeg")
    img = tmp_path / "a.jpg"
    img.write_bytes(img_path.read_bytes())
    lf = tmp_path / "list.txt"
    lf.write_text(str(img) + "\n")
    items = list(iter_list_file(lf))
    assert items[0][0] == "a"
