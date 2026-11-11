"""Stage the local launcher and its standalone demo payload; does not publish."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/extensions/com.findsurface.bootstrap"))
from findsurface.bootstrap.demo_launcher import stage_launcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True, help="fresh output folder")
    args = parser.parse_args()
    print(stage_launcher(ROOT, "com.curvsurf.findsurface_isaac_sim_launcher", args.destination))


if __name__ == "__main__":
    main()
