"""Shared test fixtures for object-harvest tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from object_harvest.schemas import ObjectItem, RunConfig


@pytest.fixture
def test_image_path() -> Path:
    """Return path to shared test image."""
    return Path("tests/data/test.jpeg")


@pytest.fixture
def test_image_data(test_image_path: Path) -> bytes:
    """Return test image data as bytes."""
    return test_image_path.read_bytes()


@pytest.fixture
def sample_objects() -> list[ObjectItem]:
    """Return sample ObjectItem instances for testing (confidence removed in new schema)."""
    return [
        ObjectItem(name="cat"),
        ObjectItem(name="dog"),
        ObjectItem(name="tree"),
    ]


@pytest.fixture
def sample_run_config(tmp_path: Path) -> RunConfig:
    """Return a basic RunConfig for testing."""
    return RunConfig(
        source_dir=tmp_path,
        output=tmp_path / "test_output.jsonl",
        model="test-model",
        threads=2,
    )


@pytest.fixture
def mock_vlm_response() -> dict[str, Any]:
    """Return a mock VLM API response (string list format)."""
    return {"objects": ["cat", "dog"]}


@pytest.fixture
def mock_boxes_response() -> dict[str, Any]:
    """Return a mock boxes API response (new dict-of-lists format)."""
    return {"cat": [[10, 15, 80, 90]], "dog": [[100, 120, 180, 200]]}


@pytest.fixture
def create_test_images(tmp_path: Path, test_image_data: bytes):
    """Create multiple test images in a directory."""

    def _create_images(names: list[str]) -> Path:
        for name in names:
            (tmp_path / f"{name}.jpg").write_bytes(test_image_data)
        return tmp_path

    return _create_images


@pytest.fixture
def create_test_jsonl():
    """Create a test JSONL file with sample data."""

    def _create_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
        with path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        return path

    return _create_jsonl
