"""Runtime discovery, pinned extension setup, and application launch."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, Sequence

REQUIRED_VERSION = (6, 0, 1)
ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE = ROOT / "source" / "apps" / "com.curvsurf.findsurface_isaac_sim_demo.kit"
EXTENSIONS = ROOT / "source" / "extensions"
BOOTSTRAP_EXTENSION = EXTENSIONS / "com.findsurface.bootstrap"

class RuntimeDiscoveryError(RuntimeError):
    """An actionable Isaac Sim runtime discovery or version error."""


def require_bootstrap() -> None:
    if not (BOOTSTRAP_EXTENSION / "config" / "extension.toml").is_file():
        raise RuntimeDiscoveryError(
            "FindSurface submodule is missing. Run repo.bat (Windows) or bash repo.sh (Ubuntu) init-dependencies "
            "(or git submodule update --init --recursive) before launch."
        )


def command_init_dependencies(_args: argparse.Namespace) -> int:
    return initialize_dependencies(ROOT)


def initialize_dependencies(project: Path) -> int:
    """Initialize the pinned extension while preserving pre-positioned SDK files."""
    project = project.resolve()
    bootstrap = project / "source/extensions/com.findsurface.bootstrap"
    backup = None
    if bootstrap.exists() and not (bootstrap / ".git").exists():
        if any(bootstrap.iterdir()):
            # Preserve existing local DLL/cache files before initializing the submodule.
            relative = bootstrap.relative_to(project).as_posix()
            tracked = subprocess.check_output(
                ["git", "-c", f"safe.directory={project.as_posix()}", "ls-files", "--stage", "--", relative],
                cwd=project, text=True)
            if not tracked.startswith("160000 "):
                raise RuntimeDiscoveryError("Refusing to move a directory that is not a pinned submodule")
            if bootstrap.resolve() != project / relative:
                raise RuntimeDiscoveryError("Refusing to move a submodule path outside the project")
            artifacts = project / ".artifacts"
            if artifacts.is_symlink():
                raise RuntimeDiscoveryError("Refusing to back up submodule files through a symlink")
            artifacts.mkdir(exist_ok=True)
            backup = Path(tempfile.mkdtemp(prefix="bootstrap-backup-", dir=artifacts)) / "bootstrap"
            shutil.move(str(bootstrap), str(backup))
            print(f"Preserved uninitialized submodule files at {backup}", flush=True)
    command = ["git", "-c", f"safe.directory={project.as_posix()}",
               "submodule", "update", "--init", "--recursive"]
    try:
        return subprocess.run(command, cwd=project, check=False).returncode
    finally:
        # A failed clone must not strand the user's SDK only in the backup folder.
        if backup is not None:
            native = backup / "data" / "native"
            if native.exists():
                destination = bootstrap / "data" / "native"
                if any(path.is_symlink() for path in (destination, *destination.parents)):
                    raise RuntimeDiscoveryError(f"Unsafe SDK restore path; original files preserved at {backup}")
                for source in native.rglob("*"):
                    target = destination / source.relative_to(native)
                    if source.is_dir() and not source.is_symlink():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        if target.exists() or target.is_symlink():
                            raise RuntimeDiscoveryError(f"Refusing to overwrite {target}; original files preserved at {backup}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target, follow_symlinks=False)
                print(f"Restored local native libraries to {destination}", flush=True)


@dataclass(frozen=True)
class RuntimeInfo:
    root: Path
    executable: Path
    apps: Path
    version: tuple[int, int, int]
    version_text: str


def parse_version(text: str) -> tuple[int, int, int]:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)", text)
    if match is None:
        raise RuntimeDiscoveryError(f"Isaac Sim version could not be parsed from {text!r}")
    return tuple(int(part) for part in match.groups())


def _read_version(root: Path) -> str:
    for name in ("VERSION", "SHORT_VERSION", "PACKAGE-INFO.yaml"):
        candidate = root / name
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace").strip()
        try:
            parse_version(text)
        except RuntimeDiscoveryError:
            continue
        return text
    raise RuntimeDiscoveryError(
        f"No parseable VERSION, SHORT_VERSION, or PACKAGE-INFO.yaml under {root}"
    )


def inspect_runtime(root: Path) -> RuntimeInfo:
    root = root.expanduser().resolve()
    executables = (
        root / "kit" / "kit.exe",
        root / "kit" / "kit",
        root / "_build" / "windows-x86_64" / "release" / "kit" / "kit.exe",
        root / "_build" / "linux-x86_64" / "release" / "kit" / "kit",
    )
    executable = next((path for path in executables if path.is_file()), None)
    if executable is None:
        raise RuntimeDiscoveryError(f"Kit executable was not found under {root}")
    app_candidates = (root / "apps", executable.parents[1] / "apps")
    apps = next((path for path in app_candidates if path.is_dir()), None)
    if apps is None:
        raise RuntimeDiscoveryError(f"Isaac Sim apps directory was not found under {root}")
    version_text = _read_version(root)
    return RuntimeInfo(root, executable, apps, parse_version(version_text), version_text.splitlines()[0])


def candidate_roots(
    explicit: str | None,
    environment: dict[str, str] | None = None,
) -> Iterable[Path]:
    environment = os.environ if environment is None else environment
    values: list[str] = []
    if explicit:
        values.append(explicit)
    if environment.get("ISAACSIM_ROOT"):
        values.append(environment["ISAACSIM_ROOT"])
    if os.name == "nt":
        values.extend((r"C:\isaacsim-6.0.1", r"C:\IsaacSim-6.0.1", r"C:\isaacsim", r"C:\IsaacSim"))
    else:
        values.append(str(Path.home() / "isaacsim"))
    seen: set[Path] = set()
    for value in values:
        resolved = Path(value).expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved


def discover_runtime(
    explicit: str | None = None,
    environment: dict[str, str] | None = None,
) -> RuntimeInfo:
    failures: list[str] = []
    for root in candidate_roots(explicit, environment):
        if not root.exists():
            failures.append(f"{root}: directory does not exist")
            continue
        try:
            return inspect_runtime(root)
        except RuntimeDiscoveryError as error:
            failures.append(str(error))
    detail = "\n  - ".join(failures) if failures else "no candidates configured"
    raise RuntimeDiscoveryError(
        "Isaac Sim runtime was not found. Pass --isaac-sim-root or set ISAACSIM_ROOT."
        f"\n  - {detail}"
    )


def require_target(runtime: RuntimeInfo) -> None:
    if runtime.version != REQUIRED_VERSION:
        actual = ".".join(map(str, runtime.version))
        required = ".".join(map(str, REQUIRED_VERSION))
        raise RuntimeDiscoveryError(
            f"Isaac Sim {required} is required; discovered {actual} at {runtime.root}."
        )


def launch_command(
    runtime: RuntimeInfo,
    extra: Sequence[str] = (),
    *,
    experience: Path = EXPERIENCE,
) -> list[str]:
    command = [str(runtime.executable), str(experience)]
    folders = (
        runtime.apps,
        runtime.root / "exts",
        runtime.root / "extscache",
        runtime.root / "extsUser",
        runtime.root / "extsDeprecated",
        EXTENSIONS,
    )
    for folder in folders:
        if folder.is_dir():
            command.extend(("--ext-folder", str(folder)))
    command.extend(extra)
    return command


def _runtime(args: argparse.Namespace) -> RuntimeInfo:
    runtime = discover_runtime(args.isaac_sim_root)
    require_target(runtime)
    return runtime


def command_doctor(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    print(f"Isaac Sim root: {runtime.root}")
    print(f"Isaac Sim version: {'.'.join(map(str, runtime.version))}")
    print(f"Kit executable: {runtime.executable}")
    print(f"Experience: {EXPERIENCE}")
    return 0


def command_launch(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    os.environ["FINDSURFACE_SDK_AUTO_DOWNLOAD"] = "0"
    forwarded = list(args.kit_args)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    return subprocess.run(
        launch_command(runtime, forwarded),
        cwd=runtime.root,
        check=False,
    ).returncode


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    children = result.add_subparsers(dest="command", required=True)
    children.add_parser("init-dependencies", help="initialize pinned Git submodules")
    for name in ("doctor", "launch"):
        child = children.add_parser(name)
        child.add_argument("--isaac-sim-root", help="exact Isaac Sim installation root")
        if name == "launch":
            child.add_argument("kit_args", nargs=argparse.REMAINDER)
    return result

def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "init-dependencies":
            return command_init_dependencies(args)
        require_bootstrap()
        if args.command == "doctor":
            return command_doctor(args)
        return command_launch(args)
    except RuntimeDiscoveryError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 3

if __name__ == "__main__":
    raise SystemExit(main())
