"""Tests for CLI functionality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from object_harvest.cli import main


class TestCLI:
    """Test cases for CLI interface."""

    def test_cli_smoke_test(
        self,
        tmp_path: Path,
        test_image_data: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test basic CLI functionality with mocked pipeline.

        Args:
            tmp_path: Temporary directory for test files
            test_image_data: Test image data fixture
            monkeypatch: Pytest monkeypatch fixture
        """
        # Prepare test image
        img = tmp_path / "test_image.png"
        img.write_bytes(test_image_data)

        # Mock pipeline to avoid network calls
        def fake_process_images(cfg) -> None:
            """Mock process_images function."""
            output_record = {
                "image_id": "test_image",
                "path": str(img),
                "model": cfg.model,
                "objects": [{"name": "cat"}],
                "boxes": [],
            }
            cfg.output.write_text(json.dumps(output_record) + "\n")

        # Patch in both pipeline and cli modules to be safe
        import object_harvest.pipeline as pipeline
        import object_harvest.cli as cli_mod

        monkeypatch.setattr(pipeline, "process_images", fake_process_images)
        monkeypatch.setattr(cli_mod, "process_images", fake_process_images)

        # Set up output file
        output_file = tmp_path / "results.jsonl"

        # Run CLI command
        argv = [
            "--source",
            str(tmp_path),
            "--model",
            "test-model",
            "--output",
            str(output_file),
        ]

        return_code = main(argv)

        # Verify results
        assert return_code == 0
        assert output_file.exists()

        content = output_file.read_text().strip()
        assert "cat" in content
        assert "test-model" in content

    @pytest.mark.parametrize(
        "source_type,expected_args",
        [
            ("folder", ["--source", "test_folder"]),
            ("list", ["--list-file", "test_list.txt"]),
        ],
    )
    def test_cli_different_source_types(
        self,
        tmp_path: Path,
        test_image_data: bytes,
        monkeypatch: pytest.MonkeyPatch,
        source_type: str,
        expected_args: list[str],
    ) -> None:
        """Test CLI with different source types.

        Args:
            tmp_path: Temporary directory
            test_image_data: Test image data
            monkeypatch: Pytest monkeypatch fixture
            source_type: Type of source (folder or list)
            expected_args: Expected CLI arguments
        """
        # Create test files based on source type
        if source_type == "folder":
            test_dir = tmp_path / "test_folder"
            test_dir.mkdir()
            (test_dir / "image.jpg").write_bytes(test_image_data)
            source_path = test_dir
        else:  # list file
            image_file = tmp_path / "image.jpg"
            image_file.write_bytes(test_image_data)
            list_file = tmp_path / "test_list.txt"
            list_file.write_text(str(image_file) + "\n")
            source_path = list_file

        # Mock pipeline
        def fake_process_images(cfg) -> None:
            cfg.output.write_text('{"objects": []}\n')

        import object_harvest.pipeline as pipeline

        monkeypatch.setattr(pipeline, "process_images", fake_process_images)

        # Run CLI with parametrized arguments
        output_file = tmp_path / "output.jsonl"
        argv = expected_args[:]  # Copy the list
        argv[1] = str(source_path)  # Replace placeholder with actual path
        argv.extend(["--model", "test", "--output", str(output_file)])

        return_code = main(argv)
        assert return_code == 0
        assert output_file.exists()
