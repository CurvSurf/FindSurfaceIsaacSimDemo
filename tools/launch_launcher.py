"""Open the local launcher UI using an already installed compatible runtime."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/extensions/com.findsurface.bootstrap"))
from findsurface.bootstrap.demo_launcher import make_launch_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", default="", help="installed runtime folder")
    parser.add_argument("kit_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.kit_args[1:] if args.kit_args[:1] == ["--"] else args.kit_args
    plan = make_launch_plan(ROOT / "source/extensions/com.curvsurf.findsurface_isaac_sim_launcher", args.runtime_root, forwarded)
    command = list(plan.command)
    command[1] = str(ROOT / "source/apps/com.curvsurf.findsurface_isaac_sim_launcher_preview.kit")
    return subprocess.call(command, cwd=plan.cwd)


if __name__ == "__main__":
    raise SystemExit(main())
