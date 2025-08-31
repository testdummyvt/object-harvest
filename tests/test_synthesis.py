import unittest
from typing import Any

import object_harvest.synthesis as synth


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Create:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **kwargs: Any):
        class _Resp:
            def __init__(self, content: str) -> None:
                self.choices = [_Choice(content)]

        return _Resp(self._content)


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None) -> None:
        self.chat = type("x", (), {})()
        self.chat.completions = _Create(self.payload)  # type: ignore[attr-defined]

    # Payload is set per test using monkeypatch-like assignment
    payload: str = ""


class TestSynthesis(unittest.TestCase):
    def setUp(self) -> None:
        # Patch OpenAI in module to use fake
        self._orig = synth.OpenAI
        synth.OpenAI = _FakeOpenAI  # type: ignore[assignment]

    def tearDown(self) -> None:
        synth.OpenAI = self._orig  # type: ignore[assignment]

    def test_strict_json_parsing(self):
        _FakeOpenAI.payload = '{"describe": "A black cat lounging on a sofa.", "objects": [{"cat": "black cat lounging"}, {"sofa": "soft fabric sofa"}]}'
        out = synth.synthesize_one_line(["cat", "sofa"], n=2, model="dummy", base_url=None)
        self.assertIn("describe", out)
        self.assertIn("objects", out)
        self.assertEqual(out["describe"], "A black cat lounging on a sofa.")
        # Ensure keys preserved and filtered
        keys = [list(d.keys())[0] for d in out["objects"]]
        self.assertEqual(keys, ["cat", "sofa"])  # in order

    def test_fallback_when_not_json(self):
        _FakeOpenAI.payload = "A bright room with a chair and a table."
        out = synth.synthesize_one_line(["chair", "table"], n=2, model="dummy", base_url=None)
        self.assertIn("describe", out)
        self.assertTrue(out["describe"].startswith("A bright room"))
        self.assertIn("objects", out)
        # Echo entries when not parsed as JSON
        self.assertEqual(out["objects"], [{"chair": "chair"}, {"table": "table"}])


if __name__ == "__main__":
    unittest.main()
