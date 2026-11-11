"""Prepare or verify a local Community archive. Never publishes."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "source/extensions/com.findsurface.bootstrap/tools/prepare_community_release.py"), run_name="__main__")
