"""The relocated preprocessing entry point imports correctly in both CLI modes."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.preprocessing import run_preprocessing

ROOT = Path(__file__).resolve().parents[1]


class PreprocessingCLITests(unittest.TestCase):
    def test_defaults_point_to_project_data_directories(self):
        self.assertEqual(run_preprocessing.PROJECT_ROOT, ROOT)
        self.assertEqual(run_preprocessing.DEFAULT_RAW_DIRECTORY, ROOT / 'data' / 'raw')
        self.assertEqual(run_preprocessing.DEFAULT_PROCESSED_DIRECTORY, ROOT / 'data' / 'processed')

    def test_help_works_as_module_and_script_outside_repository(self):
        with tempfile.TemporaryDirectory() as temporary:
            commands = [
                ([sys.executable, '-m', 'src.preprocessing.run_preprocessing', '--help'], ROOT),
                ([sys.executable, str(ROOT / 'src/preprocessing/run_preprocessing.py'), '--help'], temporary),
            ]
            for command, directory in commands:
                with self.subTest(command=command):
                    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--input-dir', result.stdout)
                    self.assertIn('--output-dir', result.stdout)


if __name__ == '__main__':
    unittest.main()
