import json
import unittest

from object_harvest.detection import parse_vlm_detections_json


class TestVLMSchemaValidation(unittest.TestCase):
    def test_happy_path(self):
        payload = {
            "detections": [
                {
                    "label": "cat",
                    "score": 0.92,
                    "bbox": {"xmin": 10, "ymin": 20, "xmax": 40, "ymax": 60},
                },
                {
                    "label": "dog",
                    "score": 1.5,
                    "bbox": {"xmin": -1, "ymin": 0, "xmax": 100, "ymax": 0},
                },
            ]
        }
        out = parse_vlm_detections_json(json.dumps(payload))
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["label"], "cat")
        self.assertAlmostEqual(out[0]["score"], 0.92, places=5)
        self.assertEqual(
            out[0]["bbox"], {"xmin": 10.0, "ymin": 20.0, "xmax": 40.0, "ymax": 60.0}
        )
        # score clamped only; pixel coords unmodified
        self.assertEqual(out[1]["label"], "dog")
        self.assertEqual(out[1]["score"], 1.0)
        self.assertEqual(
            out[1]["bbox"], {"xmin": -1.0, "ymin": 0.0, "xmax": 100.0, "ymax": 0.0}
        )

    def test_missing_required_fields(self):
        payload = {
            "detections": [
                {
                    "score": 0.5,
                    "bbox": {"x": 0, "y": 0, "w": 0.1, "h": 0.1},
                },  # no label
                {"label": "thing"},  # no bbox
                "not-an-object",  # invalid entry
            ]
        }
        out = parse_vlm_detections_json(json.dumps(payload))
        self.assertEqual(out, [])

    def test_non_object_top_level(self):
        self.assertEqual(parse_vlm_detections_json("[]"), [])
        self.assertEqual(parse_vlm_detections_json("{}"), [])

    def test_accept_string_numbers(self):
        payload = {
            "detections": [
                {
                    "label": "box",
                    "score": "0.7",
                    "bbox": {"xmin": "1", "ymin": "2", "xmax": "3", "ymax": "4"},
                },
            ]
        }
        out = parse_vlm_detections_json(json.dumps(payload))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["score"], 0.7)
        self.assertEqual(
            out[0]["bbox"], {"xmin": 1.0, "ymin": 2.0, "xmax": 3.0, "ymax": 4.0}
        )

    def test_empty_and_malformed(self):
        payload = {"detections": []}
        self.assertEqual(parse_vlm_detections_json(json.dumps(payload)), [])
        self.assertEqual(parse_vlm_detections_json("not-json"), [])
        payload2 = {
            "detections": [
                {
                    "label": "a",
                    "bbox": {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0},
                    "extra": 123,
                }
            ]
        }
        out = parse_vlm_detections_json(json.dumps(payload2))
        self.assertEqual(len(out), 1)
        self.assertNotIn("extra", out[0])


if __name__ == "__main__":
    unittest.main()
