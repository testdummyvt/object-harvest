"""Tests for pipeline functionality with mocked VLM clients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from object_harvest.pipeline import process_images
from object_harvest.schemas import RunConfig

if TYPE_CHECKING:
    from object_harvest.schemas import ObjectItem, BoxItem


class MockVLMClient:
    """Mock VLM client for testing pipeline without network calls."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize mock client (ignores all arguments)."""
        pass

    def list_objects(self, image_path: Path) -> tuple[list[ObjectItem], str | None]:
        """Return mock objects for any image.

        Args:
            image_path: Path to image file (ignored in mock)

        Returns:
            Tuple of mock ObjectItem list and no error
        """
        from object_harvest.schemas import ObjectItem

        return [
            ObjectItem(name="cat"),
            ObjectItem(name="dog"),
        ], None

    def list_boxes(
        self,
        image_path: Path,
        object_list,
        size: tuple[int, int] | None = None,
    ) -> tuple[list[BoxItem], str | None]:
        """Return mock bounding boxes for objects.

        Args:
            image_path: Path to image file (ignored in mock)
            object_list: List of objects to create boxes for
            size: Image size (ignored in mock)

        Returns:
            Tuple of mock BoxItem list and no error
        """
        from object_harvest.schemas import BoxItem

        return [
            BoxItem(name="cat", x1=10, y1=15, x2=80, y2=90),
            BoxItem(name="dog", x1=100, y1=120, x2=180, y2=200),
        ], None


class TestPipelineWithMock:
    """Test cases for pipeline functionality using mocked VLM client."""

    def test_pipeline_processes_single_image(
        self,
        tmp_path: Path,
        test_image_data: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test pipeline processing a single image.

        Args:
            tmp_path: Temporary directory fixture
            test_image_data: Test image data fixture
            monkeypatch: Pytest monkeypatch fixture
        """
        # Create test image
        test_image = tmp_path / "test_image.jpg"
        test_image.write_bytes(test_image_data)

        # Mock VLMClient
        import object_harvest.pipeline as pipeline

        monkeypatch.setattr(pipeline, "VLMClient", MockVLMClient)

        # Configure pipeline run
        output_file = tmp_path / "results.jsonl"
        config = RunConfig(
            source_dir=tmp_path,
            output=output_file,
            model="mock-model",
            boxes=False,
            threads=1,
        )

        # Run pipeline
        process_images(config)

        # Verify output
        assert output_file.exists()
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 1

        result = json.loads(lines[0])
        assert result["image_id"] == "test_image"
        assert result["model"] == "mock-model"
        assert len(result["objects"]) == 2
        assert any(obj["name"] == "cat" for obj in result["objects"])
        assert any(obj["name"] == "dog" for obj in result["objects"])

    def test_pipeline_with_bounding_boxes(
        self,
        tmp_path: Path,
        test_image_data: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test pipeline with bounding box generation enabled.

        Args:
            tmp_path: Temporary directory fixture
            test_image_data: Test image data fixture
            monkeypatch: Pytest monkeypatch fixture
        """
        # Create test image
        test_image = tmp_path / "image_with_boxes.jpg"
        test_image.write_bytes(test_image_data)

        # Mock VLMClient
        import object_harvest.pipeline as pipeline

        monkeypatch.setattr(pipeline, "VLMClient", MockVLMClient)

        # Configure pipeline run with boxes enabled
        output_file = tmp_path / "results_with_boxes.jsonl"
        config = RunConfig(
            source_dir=tmp_path,
            output=output_file,
            model="mock-model",
            boxes=True,
            threads=1,
        )

        # Run pipeline
        process_images(config)

        # Verify output includes boxes
        assert output_file.exists()
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 1

        result = json.loads(lines[0])
        assert len(result["boxes"]) == 2

        # Check box structure
        for box in result["boxes"]:
            assert "name" in box
            assert "x1" in box and "y1" in box
            assert "x2" in box and "y2" in box
            assert box["x1"] < box["x2"]
            assert box["y1"] < box["y2"]

    def test_pipeline_processes_multiple_images(
        self,
        tmp_path: Path,
        test_image_data: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test pipeline processing multiple images.

        Args:
            tmp_path: Temporary directory fixture
            test_image_data: Test image data fixture
            monkeypatch: Pytest monkeypatch fixture
        """
        # Create multiple test images
        image_names = ["image1", "image2", "image3"]
        for name in image_names:
            (tmp_path / f"{name}.jpg").write_bytes(test_image_data)

        # Mock VLMClient
        import object_harvest.pipeline as pipeline

        monkeypatch.setattr(pipeline, "VLMClient", MockVLMClient)

        # Configure pipeline run
        output_file = tmp_path / "multi_results.jsonl"
        config = RunConfig(
            source_dir=tmp_path,
            output=output_file,
            model="mock-model",
            threads=2,  # Test with multiple threads
        )

        # Run pipeline
        process_images(config)

        # Verify output
        assert output_file.exists()
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 3

        # Parse all results
        results = [json.loads(line) for line in lines]
        processed_ids = {result["image_id"] for result in results}

        assert processed_ids == set(image_names)

        # Verify each result has expected structure
        for result in results:
            assert result["model"] == "mock-model"
            assert len(result["objects"]) == 2
            assert "t_total" in result
            assert result["attempts"] == 1

    @pytest.mark.parametrize("thread_count", [1, 2, 4])
    def test_pipeline_thread_safety(
        self,
        tmp_path: Path,
        test_image_data: bytes,
        monkeypatch: pytest.MonkeyPatch,
        thread_count: int,
    ) -> None:
        """Test pipeline with different thread counts.

        Args:
            tmp_path: Temporary directory fixture
            test_image_data: Test image data fixture
            monkeypatch: Pytest monkeypatch fixture
            thread_count: Number of threads to use
        """
        # Create test images
        num_images = 5
        for i in range(num_images):
            (tmp_path / f"img_{i}.jpg").write_bytes(test_image_data)

        # Mock VLMClient
        import object_harvest.pipeline as pipeline

        monkeypatch.setattr(pipeline, "VLMClient", MockVLMClient)

        # Configure pipeline run
        output_file = tmp_path / f"thread_test_{thread_count}.jsonl"
        config = RunConfig(
            source_dir=tmp_path,
            output=output_file,
            model="mock-model",
            threads=thread_count,
        )

        # Run pipeline
        process_images(config)

        # Verify all images were processed
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == num_images
