import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(os.environ.get("COMFYUI_ROOT"), "Set COMFYUI_ROOT for the real ComfyUI integration test")
class ComfyUIIntegrationTests(unittest.TestCase):
    def test_real_upscale_node_in_fresh_process(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("comfyui_smoke.py"))],
            capture_output=True, text=True, encoding="utf-8", timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
