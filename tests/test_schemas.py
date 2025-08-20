from object_harvest.schemas import ObjectItem, safe_parse_objects


def test_safe_parse_objects_basic():
    raw = '{"objects": [{"name": "cat", "confidence": 0.9}, {"name": "dog"}]}'
    objs, err = safe_parse_objects(raw)
    assert err is None
    assert [o.name for o in objs] == ["cat", "dog"]


def test_safe_parse_objects_markdown_fence():
    raw = """```json\n{"objects": [{"name": "tree"}]}\n```"""
    objs, err = safe_parse_objects(raw)
    assert err is None
    assert objs[0].name == "tree"
