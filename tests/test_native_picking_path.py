import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/extensions/com.curvsurf.findsurface_isaac_sim"))
from findsurface_isaac_sim.native_picking import default_library_path


class NativePickingPathTests(unittest.TestCase):
    def test_private_launch_override_preserves_spaces(self):
        expected = (ROOT / ".artifacts/test picker/private.so").resolve()
        with patch.dict(os.environ, {"FSIS_PICKING_LIBRARY_PATH": str(expected)}):
            self.assertEqual(default_library_path(), expected)

    def test_unset_override_uses_packaged_platform_library(self):
        with patch.dict(os.environ, {"FSIS_PICKING_LIBRARY_PATH": ""}):
            path = default_library_path()
            self.assertEqual(path.parent.name, "windows-x86_64" if os.name == "nt" else "linux-x86_64")


if __name__ == "__main__":
    unittest.main()
