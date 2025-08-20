"""Tests for schema validation and parsing functionality."""

from __future__ import annotations

import pytest

from object_harvest.schemas import ObjectItem, safe_parse_objects


class TestObjectItem:
    """Test cases for ObjectItem schema validation."""

    def test_object_item_valid_creation(self) -> None:
        """Test creating valid ObjectItem instances."""
        # Test with confidence
        obj_with_conf = ObjectItem(name="cat", confidence=0.9)
        assert obj_with_conf.name == "cat"
        assert obj_with_conf.confidence == 0.9

        # Test without confidence
        obj_without_conf = ObjectItem(name="dog")
        assert obj_without_conf.name == "dog"
        assert obj_without_conf.confidence is None

    @pytest.mark.parametrize(
        "confidence,expected_valid",
        [
            (0.0, True),
            (0.5, True),
            (1.0, True),
            (-0.1, False),
            (1.1, False),
            (None, True),
        ],
    )
    def test_object_item_confidence_validation(
        self,
        confidence: float | None,
        expected_valid: bool,
    ) -> None:
        """Test confidence value validation.

        Args:
            confidence: Confidence value to test
            expected_valid: Whether the value should be valid
        """
        if expected_valid:
            obj = ObjectItem(name="test", confidence=confidence)
            assert obj.confidence == confidence
        else:
            with pytest.raises(ValueError):
                ObjectItem(name="test", confidence=confidence)

    def test_object_item_name_required(self) -> None:
        """Test that name field is required."""
        with pytest.raises(ValueError):
            ObjectItem()  # type: ignore[call-arg]


class TestSafeParseObjects:
    """Test cases for safe_parse_objects function."""

    def test_parse_valid_json_objects(self) -> None:
        """Test parsing valid JSON with objects array."""
        json_input = '{"objects": [{"name": "cat", "confidence": 0.9}, {"name": "dog"}]}'
        objects, error = safe_parse_objects(json_input)

        assert error is None
        assert len(objects) == 2
        assert objects[0].name == "cat"
        assert objects[0].confidence == 0.9
        assert objects[1].name == "dog"
        assert objects[1].confidence is None

    def test_parse_json_with_markdown_fences(self) -> None:
        """Test parsing JSON wrapped in markdown code fences."""
        json_input = '```json\n{"objects": [{"name": "tree"}]}\n```'
        objects, error = safe_parse_objects(json_input)

        assert error is None
        assert len(objects) == 1
        assert objects[0].name == "tree"

    @pytest.mark.parametrize(
        "fence_type",
        [
            "```json",
            "```JSON",
            "```",
        ],
    )
    def test_parse_different_markdown_fences(self, fence_type: str) -> None:
        """Test parsing with different markdown fence styles.

        Args:
            fence_type: Type of markdown fence to test
        """
        json_content = '{"objects": [{"name": "bird"}]}'
        json_input = f"{fence_type}\n{json_content}\n```"

        objects, error = safe_parse_objects(json_input)

        assert error is None
        assert len(objects) == 1
        assert objects[0].name == "bird"

    def test_parse_direct_array(self) -> None:
        """Test parsing JSON that's directly an array of objects."""
        json_input = '[{"name": "cat"}, {"name": "dog"}]'
        objects, error = safe_parse_objects(json_input)

        assert error is None
        assert len(objects) == 2
        assert objects[0].name == "cat"
        assert objects[1].name == "dog"

    @pytest.mark.parametrize("name_field", ["name", "label", "object"])
    def test_parse_different_name_fields(self, name_field: str) -> None:
        """Test parsing objects with different name field variations.

        Args:
            name_field: The field name to use for object names
        """
        json_input = f'{{"objects": [{{{name_field}: "test_object"}}]}}'
        objects, error = safe_parse_objects(json_input)

        assert error is None
        assert len(objects) == 1
        assert objects[0].name == "test_object"

    @pytest.mark.parametrize("confidence_field", ["confidence", "score"])
    def test_parse_different_confidence_fields(self, confidence_field: str) -> None:
        """Test parsing objects with different confidence field names.

        Args:
            confidence_field: The field name to use for confidence values
        """
        json_input = f'{{"objects": [{{"name": "test", "{confidence_field}": 0.8}}]}}'
        objects, error = safe_parse_objects(json_input)

        assert error is None
        assert len(objects) == 1
        assert objects[0].confidence == 0.8

    def test_parse_ignores_invalid_items(self) -> None:
        """Test that parser ignores invalid items but processes valid ones."""
        json_input = """
        {
            "objects": [
                {"name": "valid_cat", "confidence": 0.9},
                {"no_name_field": "invalid"},
                null,
                "not_an_object",
                {"name": "valid_dog"}
            ]
        }
        """

        objects, error = safe_parse_objects(json_input)

        assert error is None
        assert len(objects) == 2
        assert objects[0].name == "valid_cat"
        assert objects[1].name == "valid_dog"

    def test_parse_malformed_json_returns_error(self) -> None:
        """Test that malformed JSON returns an error."""
        malformed_inputs = [
            '{"objects": [{"name": "incomplete"',  # Missing closing braces
            "not json at all",
            '{"objects": invalid_json}',
            "",
        ]

        for malformed_input in malformed_inputs:
            objects, error = safe_parse_objects(malformed_input)
            assert error is not None
            assert objects == []

    def test_parse_non_list_data_returns_error(self) -> None:
        """Test that non-list data returns appropriate error."""
        json_input = '{"objects": "not_a_list"}'
        objects, error = safe_parse_objects(json_input)

        assert error == "data_not_list"
        assert objects == []

    def test_parse_empty_objects_array(self) -> None:
        """Test parsing empty objects array."""
        json_input = '{"objects": []}'
        objects, error = safe_parse_objects(json_input)

        assert error is None
        assert objects == []

    def test_parse_mixed_valid_invalid_confidence_values(self) -> None:
        """Test parsing objects with mix of valid and invalid confidence values."""
        json_input = """
        {
            "objects": [
                {"name": "obj1", "confidence": 0.9},
                {"name": "obj2", "confidence": "invalid_string"},
                {"name": "obj3", "confidence": 1.5},
                {"name": "obj4"}
            ]
        }
        """

        objects, error = safe_parse_objects(json_input)

        # Should process valid objects, skip invalid confidence values
        assert error is None
        valid_names = [obj.name for obj in objects]
        assert "obj1" in valid_names
        assert "obj4" in valid_names

        # Check that valid confidence is preserved
        obj1 = next(obj for obj in objects if obj.name == "obj1")
        assert obj1.confidence == 0.9
