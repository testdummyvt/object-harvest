"""Tests for image ingestion functionality."""

from __future__ import annotations

from pathlib import Path

import pytest

from object_harvest.ingest import iter_folder, iter_list_file


class TestIngestion:
    """Test cases for image ingestion utilities."""

    @pytest.mark.parametrize(
        "extensions,expected_count",
        [
            ([".jpg", ".png"], 2),
            ([".jpg", ".png", ".webp"], 3),
            ([".txt"], 0),  # Text files should be ignored
            ([".JPG", ".PNG"], 2),  # Test case sensitivity
        ],
    )
    def test_iter_folder_filters_by_extension(
        self,
        tmp_path: Path,
        test_image_data: bytes,
        extensions: list[str],
        expected_count: int,
    ) -> None:
        """Test that iter_folder correctly filters files by image extensions.

        Args:
            tmp_path: Temporary directory fixture
            test_image_data: Test image data fixture
            extensions: File extensions to create
            expected_count: Expected number of valid images
        """
        # Create test files with various extensions
        for i, ext in enumerate(extensions):
            if ext in {".jpg", ".png", ".webp", ".JPG", ".PNG"}:
                (tmp_path / f"image_{i}{ext}").write_bytes(test_image_data)
            else:
                (tmp_path / f"file_{i}{ext}").write_text("not an image")

        # Test the function
        items = list(iter_folder(tmp_path))
        assert len(items) == expected_count

        # Verify all returned items are valid image files
        for _image_id, path in items:
            assert path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            assert path.exists()

    def test_iter_folder_returns_sorted_results(
        self,
        tmp_path: Path,
        test_image_data: bytes,
    ) -> None:
        """Test that iter_folder returns results in sorted order.

        Args:
            tmp_path: Temporary directory fixture
            test_image_data: Test image data fixture
        """
        # Create images with names that would be out of order without sorting
        image_names = ["z_image", "a_image", "m_image"]
        for name in image_names:
            (tmp_path / f"{name}.jpg").write_bytes(test_image_data)

        items = list(iter_folder(tmp_path))
        returned_names = [image_id for image_id, _ in items]

        assert returned_names == sorted(image_names)

    def test_iter_folder_handles_subdirectories(
        self,
        tmp_path: Path,
        test_image_data: bytes,
    ) -> None:
        """Test that iter_folder recursively finds images in subdirectories.

        Args:
            tmp_path: Temporary directory fixture
            test_image_data: Test image data fixture
        """
        # Create nested directory structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        nested_subdir = subdir / "nested"
        nested_subdir.mkdir()

        # Create images at different levels
        (tmp_path / "root.jpg").write_bytes(test_image_data)
        (subdir / "sub.png").write_bytes(test_image_data)
        (nested_subdir / "nested.webp").write_bytes(test_image_data)

        items = list(iter_folder(tmp_path))
        names = sorted(image_id for image_id, _ in items)

        assert names == ["nested", "root", "sub"]

    def test_iter_list_file_basic_functionality(
        self,
        tmp_path: Path,
        test_image_data: bytes,
    ) -> None:
        """Test basic functionality of iter_list_file.

        Args:
            tmp_path: Temporary directory fixture
            test_image_data: Test image data fixture
        """
        # Create test images
        img1 = tmp_path / "image1.jpg"
        img2 = tmp_path / "image2.png"
        img1.write_bytes(test_image_data)
        img2.write_bytes(test_image_data)

        # Create list file
        list_file = tmp_path / "images.txt"
        list_file.write_text(f"{img1}\n{img2}\n")

        items = list(iter_list_file(list_file))

        assert len(items) == 2
        assert items[0][0] == "image1"
        assert items[1][0] == "image2"
        assert str(items[0][1]) == str(img1)
        assert str(items[1][1]) == str(img2)

    def test_iter_list_file_handles_empty_lines(
        self,
        tmp_path: Path,
        test_image_data: bytes,
    ) -> None:
        """Test that iter_list_file correctly handles empty lines and whitespace.

        Args:
            tmp_path: Temporary directory fixture
            test_image_data: Test image data fixture
        """
        # Create test image
        img = tmp_path / "test.jpg"
        img.write_bytes(test_image_data)

        # Create list file with empty lines and whitespace
        list_file = tmp_path / "list.txt"
        list_content = f"""
        
{img}

        
        """
        list_file.write_text(list_content)

        items = list(iter_list_file(list_file))

        # Should only return one item, ignoring empty lines
        assert len(items) == 1
        assert items[0][0] == "test"

    @pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
    def test_iter_list_file_handles_different_line_endings(
        self,
        tmp_path: Path,
        test_image_data: bytes,
        line_ending: str,
    ) -> None:
        """Test that iter_list_file handles different line endings.

        Args:
            tmp_path: Temporary directory fixture
            test_image_data: Test image data fixture
            line_ending: Type of line ending to test
        """
        # Create test images
        img1 = tmp_path / "a.jpg"
        img2 = tmp_path / "b.jpg"
        img1.write_bytes(test_image_data)
        img2.write_bytes(test_image_data)

        # Create list file with specific line endings
        list_file = tmp_path / "list.txt"
        content = f"{img1}{line_ending}{img2}{line_ending}"
        list_file.write_bytes(content.encode())

        items = list(iter_list_file(list_file))

        assert len(items) == 2
        names = [item[0] for item in items]
        assert names == ["a", "b"]
