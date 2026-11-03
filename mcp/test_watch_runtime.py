import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))
import watch_runtime


class StreamStateTests(unittest.TestCase):
    def test_cursor_advances_and_resets_on_session_change(self):
        state = watch_runtime.StreamState()
        first, reset = state.accept({"session_id": "one", "samples": []})
        self.assertEqual(first, [])
        self.assertTrue(reset)
        self.assertEqual(state.cursor, 0)

        samples, reset = state.accept({
            "session_id": "one",
            "samples": [{"sequence": 1}, {"sequence": 3}, {"sequence": 2}],
        })
        self.assertFalse(reset)
        self.assertEqual([item["sequence"] for item in samples], [1, 3])
        self.assertEqual(state.cursor, 3)

        samples, reset = state.accept({"session_id": "two", "samples": [{"sequence": 9}]})
        self.assertTrue(reset)
        self.assertEqual(samples, [])
        self.assertEqual(state.cursor, 0)


class RotatingJsonlTests(unittest.TestCase):
    def test_rotation_keeps_at_most_configured_files_and_size(self):
        with tempfile.TemporaryDirectory(dir="E:/") as temp_dir:
            directory = Path(temp_dir)
            with mock.patch.object(watch_runtime, "ROTATE_BYTES", 400):
                writer = watch_runtime.RotatingJsonl(directory)
                for index in range(30):
                    writer.write({"index": index, "padding": "x" * 100})
            active = directory / "runtime-metrics.jsonl"
            files = sorted(
                directory.glob("runtime-metrics.jsonl*"),
                key=lambda path: 0 if path == active else -int(path.name.rsplit(".", 1)[1]),
            )
            self.assertLessEqual(len(files), watch_runtime.ROTATE_FILES)
            self.assertTrue(all(path.stat().st_size <= 400 for path in files))
            records = [json.loads(line) for path in files for line in path.read_text().splitlines()]
            self.assertTrue(records)
            self.assertEqual(records[-1]["index"], 29)


class ArgumentValidationTests(unittest.TestCase):
    def test_rejects_c_drive_and_limits_duration(self):
        with self.assertRaisesRegex(ValueError, "E:"):
            watch_runtime.require_e_path("C:/private/output", "--output-dir", must_exist=False)
        for value in (0, -1, float('nan'), float('inf'), watch_runtime.MAX_DURATION_SECONDS + 1):
            with self.assertRaises(ValueError):
                watch_runtime.validate_duration(value)
        watch_runtime.validate_duration(None)
        watch_runtime.validate_duration(watch_runtime.MAX_DURATION_SECONDS)


if __name__ == "__main__":
    unittest.main()
