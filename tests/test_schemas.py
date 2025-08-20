"""Tests for schema validation and parsing functionality (updated schema)."""

from __future__ import annotations

import json
import pytest

from object_harvest.schemas import ObjectItem, safe_parse_objects


class TestObjectItem:
    def test_object_item_valid_creation(self) -> None:
        assert ObjectItem(name="cat").name == "cat"

    def test_object_item_name_required(self) -> None:
        with pytest.raises(ValueError):  # type: ignore[arg-type]
            ObjectItem()  # type: ignore[call-arg]


class TestSafeParseObjects:
    def test_parse_objects_dict_format(self) -> None:
        data = {"objects": [{"name": "cat"}, {"name": "dog"}]}
        objects, err = safe_parse_objects(json.dumps(data))
        assert err is None
        assert [o.name for o in objects] == ["cat", "dog"]

    def test_parse_objects_string_list(self) -> None:
        data = {"objects": ["cat", "dog", "tree"]}
        objects, err = safe_parse_objects(json.dumps(data))
        assert err is None
        assert [o.name for o in objects] == ["cat", "dog", "tree"]

    def test_parse_markdown_fenced_json(self) -> None:
        payload = json.dumps({"objects": ["bird"]})
        text = f"```json\n{payload}\n```"
        objects, err = safe_parse_objects(text)
        assert err is None
        assert [o.name for o in objects] == ["bird"]

    def test_parse_various_fence_types(self) -> None:
        payload = json.dumps({"objects": ["owl"]})
        for fence in ["```json", "```JSON", "```"]:
            text = f"{fence}\n{payload}\n```"
            objs, err = safe_parse_objects(text)
            assert err is None
            assert [o.name for o in objs] == ["owl"]

    @pytest.mark.parametrize("field", ["name", "label", "object"])
    def test_parse_alternate_name_fields(self, field: str) -> None:
        # Build JSON with alternate key
        data = {"objects": [{field: "thing"}]}
        objects, err = safe_parse_objects(json.dumps(data))
        assert err is None
        assert len(objects) == 1 and objects[0].name == "thing"

    def test_parse_ignores_invalid_items(self) -> None:
        data = {
            "objects": [
                {"name": "valid_cat"},
                {"no_name_field": "invalid"},
                None,
                123,
                {"label": "valid_dog"},
            ]
        }
        objects, err = safe_parse_objects(json.dumps(data))
        assert err is None
        assert [o.name for o in objects] == ["valid_cat", "valid_dog"]

    def test_parse_malformed_json(self) -> None:
        bad_inputs = [
            '{"objects": [{"name": "incomplete"',
            "not json",
            '{"objects": invalid}',
            "",
        ]
        for text in bad_inputs:
            objs, err = safe_parse_objects(text)
            assert err is not None
            assert objs == []

    def test_parse_non_list_objects(self) -> None:
        objs, err = safe_parse_objects('{"objects": "not_a_list"}')
        assert err == "data_not_list"
        assert objs == []

    def test_parse_empty_list(self) -> None:
        objs, err = safe_parse_objects('{"objects": []}')
        assert err is None and objs == []
