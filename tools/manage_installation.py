"""Ownership-aware Ubuntu setup, launch and removal. Windows uses the PS1 peer."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import uuid
import zipfile

from linux_archive import extract_archive
import repo
import update_sdk

APP = "FindSurfaceIsaacSimDemo"
VERSION = "6.0.1"
ROOT = Path(__file__).resolve().parents[1]
PICKER_NAME = "libFindSurfaceIsaacSimPicking.so"
SDK_NAME = "libFindSurface.so"
SETTING = "--/exts/com.curvsurf.findsurface_isaac_sim/findSurfaceLibraryPath="


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    no_symlink_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = no_symlink_path(path.with_suffix(".tmp"))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def no_symlink_path(path: Path) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Symbolic link is not allowed in an ownership path: {path}")
    return path


def checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def receive(url: str, destination: Path, expected: str, algorithm: str = "sha256") -> None:
    destination = no_symlink_path(destination)
    length = 64 if algorithm == "sha256" else 32
    if not url.startswith("https://") or not re.fullmatch(rf"[a-fA-F0-9]{{{length}}}", expected):
        raise ValueError("Download requires HTTPS and a pinned checksum")
    if destination.is_file():
        if checksum(destination, algorithm) != expected.lower():
            raise ValueError(f"Cached file checksum mismatch; retained for inspection: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = no_symlink_path(destination.with_suffix(destination.suffix + ".partial"))
    subprocess.run(["curl", "--fail", "--location", "--proto", "=https", "--proto-redir",
                    "=https", "--retry", "3", "--continue-at", "-", "--output", str(partial), url], check=True)
    print(f"Verifying {algorithm.upper()}: {partial.name}", flush=True)
    if checksum(partial, algorithm) != expected.lower():
        raise ValueError(f"Download checksum mismatch; retained for inspection: {partial}")
    partial.replace(destination)


class Installation:
    def __init__(self, project: Path = ROOT):
        self.project = project.resolve()
        self.record_path = self.project / ".installation/manifest.json"

    @contextmanager
    def lock(self):
        import fcntl
        no_symlink_path(self.record_path)
        self.record_path.parent.mkdir(exist_ok=True)
        lock_path = no_symlink_path(self.record_path.with_name("operation.lock"))
        with lock_path.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("This checkout is already running or being prepared/removed") from error
            yield

    def owned_root(self, record: dict) -> Path:
        identifier = str(uuid.UUID(record["id"]))
        root = no_symlink_path(Path(record["private_root"]))
        if (record.get("schema") != 1 or record.get("app") != APP
                or record.get("platform") != "linux-x86_64"
                or record.get("project_root") != str(self.project)
                or root.name != f"fsis-{identifier}" or not root.is_dir()):
            raise ValueError("Installation ownership/checkout identity mismatch")
        marker_path = no_symlink_path(root / ".fsis-owner.json")
        marker = read_json(marker_path)
        if marker != {"app": APP, "id": identifier, "project_root": str(self.project)}:
            raise ValueError("Private directory owner marker mismatch")
        return root

    def create(self, base: Path) -> dict:
        base = no_symlink_path(base)
        base.mkdir(parents=True, exist_ok=True)
        identifier = str(uuid.uuid4())
        root = base / f"fsis-{identifier}"
        root.mkdir()
        marker = {"app": APP, "id": identifier, "project_root": str(self.project)}
        write_json(root / ".fsis-owner.json", marker)
        record = dict(marker, schema=1, platform="linux-x86_64", private_root=str(root))
        self.save(record)
        return record

    def save(self, record: dict) -> None:
        write_json(self.record_path, record)

    def remove(self, record: dict, confirm=input) -> None:
        root = self.owned_root(record)
        print(f"Permanently remove ONLY this private directory:\n  {root}")
        for key in ("runtime", "sdk", "picker"):
            item = record.get(key, {})
            if item.get("ownership") == "external":
                print(f"Preserve external {key}: {item['path']}")
        print("App source, exports, user settings and shared NVIDIA caches are preserved.")
        if confirm("Type REMOVE to confirm: ") != "REMOVE":
            print("Removal cancelled.")
            return
        for process in Path("/proc").glob("[0-9]*/comm"):
            try:
                if process.read_text().strip().lower() in ("kit", "kit.exe"):
                    raise ValueError("Close all Kit/Isaac Sim apps before removal")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        root = self.owned_root(record)
        # Linux shutil.rmtree is fd-based and unlinks child symlinks, not their targets.
        if not shutil.rmtree.avoids_symlink_attacks:
            raise ValueError("Safe fd-based recursive removal is unavailable on this Python")
        shutil.rmtree(root)
        self.record_path.unlink()
        print("App-private dependencies permanently removed. External files were preserved.")


def default_base() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "CurvSurf" / APP


def runtime_python(root: Path) -> Path:
    # NVIDIA's wrapper supplies the bundled Python shared-library search paths.
    return root / "python.sh"


def valid_runtime(root: Path) -> bool:
    try:
        runtime = repo.inspect_runtime(root)
        repo.require_target(runtime)
        return (runtime.executable.name == "kit" and os.access(runtime.executable, os.X_OK)
                and os.access(root / "kit/python/bin/python3", os.X_OK)
                and os.access(runtime_python(root), os.X_OK))
    except (repo.RuntimeDiscoveryError, OSError):
        return False


def find_runtime(explicit: str | None, record: dict, private: Path) -> Path | None:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not valid_runtime(root):
            raise ValueError(f"Explicit runtime is not complete Linux Isaac Sim {VERSION}: {root}")
        return root
    candidates = [Path(record.get("runtime", {}).get("path", private / f"isaacsim-{VERSION}"))]
    candidates.extend(repo.candidate_roots(None))
    candidates.extend((Path.home() / f"isaacsim-{VERSION}", Path("/opt/isaacsim")))
    for base in {default_base(), private.parent}:
        candidates.extend(base.glob(f"fsis-*/isaacsim-{VERSION}"))
    for candidate in candidates:
        if candidate.is_dir() and valid_runtime(candidate):
            return candidate.resolve()
    return None


def install_runtime(root: Path, config: dict) -> Path:
    target = root / f"isaacsim-{VERSION}"
    if target.exists() or target.is_symlink():
        raise ValueError(f"Refusing to overwrite an incomplete runtime: {target}")
    if shutil.disk_usage(root).free < 50 * 2**30:
        raise ValueError("At least 50 GiB free space is required for download and extraction")
    print("Isaac Sim download is approximately 13 GB (12.1 GiB); extraction also takes several minutes.")
    print("Review https://docs.isaacsim.omniverse.nvidia.com/6.0.1/common/licenses-isaac-sim.html")
    if input("Type YES to accept those terms and install: ") != "YES":
        raise ValueError("Installation cancelled")
    archive = root / "downloads/isaacsim.zip"
    receive(config["isaac_url"], archive, config["isaac_md5"], "md5")
    staging = root / f"extract-{uuid.uuid4()}"
    try:
        extract_archive(archive, staging)
        if not valid_runtime(staging):
            raise ValueError("Extracted runtime failed executable/Python/apps/version validation")
        staging.rename(target)
    except Exception:
        print(f"Extraction retained for diagnosis: {staging}", file=sys.stderr)
        raise
    return target


def record_library(record: dict, key: str, path: Path, private_path: Path) -> None:
    record[key] = {"path": str(path), "sha256": checksum(path),
                   "ownership": "private" if path == private_path else "external"}


def recorded_library(record: dict, key: str, private_path: Path) -> Path | None:
    item = record.get(key)
    if not item:
        return None
    path = Path(item["path"])
    if item.get("ownership") == "private":
        if path != private_path or path.is_symlink():
            raise ValueError(f"Private {key} path mismatch")
        if path.is_file() and checksum(path) != item["sha256"]:
            policy = read_json(update_sdk.POLICY_PATH) if key == "sdk" else {"releases": []}
            approved = {entry["assets"]["linux-x86_64"]["sha256"] for entry in policy["releases"]}
            if checksum(path) not in approved:
                raise ValueError(f"Private {key} checksum mismatch")
            print("Recognized an approved SDK replacement; refreshing its ownership digest.")
    return path if path.is_file() else None


def prepare_picker(project: Path, private: Path, record: dict) -> Path:
    packaged = project / "source/extensions/com.curvsurf.findsurface_isaac_sim/data/native/linux-x86_64" / PICKER_NAME
    if packaged.is_file():
        return packaged.resolve()
    source = project / "source/native"
    source_hash = hashlib.sha256(b"".join((source / name).read_bytes()
                                        for name in ("CMakeLists.txt", "FindSurfacePicking.cpp"))).hexdigest()
    built = no_symlink_path(private / "picker-build" / PICKER_NAME)
    existing = recorded_library(record, "picker", built)
    if existing and record.get("picker_source_sha256") == source_hash:
        return existing
    subprocess.run(["cmake", "-S", str(source), "-B", str(built.parent),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(built.parent), "--config", "Release", "--parallel", "2"], check=True)
    if not built.is_file():
        raise ValueError("Native picker build did not produce the expected shared library")
    record["picker_source_sha256"] = source_hash
    return built


def start(args: argparse.Namespace, installation: Installation) -> int:
    project = installation.project
    config = read_json(project / "tools/installation_dependencies.json")["linux-x86_64"]
    record = read_json(installation.record_path) if installation.record_path.exists() else None
    private = installation.owned_root(record) if record else None
    local_sdk = project / "source/extensions/com.findsurface.bootstrap/data/native/linux-x86_64" / SDK_NAME
    sdk = Path(args.sdk_path).expanduser().resolve() if args.sdk_path else None
    if sdk and not sdk.is_file():
        raise ValueError(f"Explicit SDK does not exist: {sdk}")
    if sdk is None and record:
        sdk = recorded_library(record, "sdk", private / "sdk" / SDK_NAME)
    sdk = sdk or (local_sdk.resolve() if local_sdk.is_file() else None)
    packaged_picker = project / "source/extensions/com.curvsurf.findsurface_isaac_sim/data/native/linux-x86_64" / PICKER_NAME
    if not packaged_picker.is_file() and (not shutil.which("cmake") or not shutil.which("c++")):
        raise ValueError("Native picker needs CMake and a C++ compiler. Run: sudo apt install cmake build-essential")
    if not shutil.which("curl"):
        raise ValueError("curl is required for verified resumable downloads. Run: sudo apt install curl")
    bootstrap = project / "source/extensions/com.findsurface.bootstrap/config/extension.toml"
    if not bootstrap.is_file():
        if not (project / ".git").exists() or not shutil.which("git"):
            raise ValueError("Missing extension source. Use a recursive Git clone or an assembled release ZIP.")
        if repo.initialize_dependencies(project) != 0:
            raise ValueError("Pinned submodule initialization failed; local SDK files were preserved. "
                             "Resolve Git/network access and rerun Start_Demo.sh.")
        if not bootstrap.is_file():
            raise ValueError("Pinned extension initialization did not supply its config")
    if not record:
        base = args.install_base
        if not base:
            base = input(f"Private dependency base folder (Enter for {default_base()}): ") or str(default_base())
        record = installation.create(Path(base))
        private = installation.owned_root(record)
    runtime = find_runtime(args.isaac_root, record, private) or install_runtime(private, config)
    record["runtime"] = {"path": str(runtime), "version": VERSION,
                         "ownership": "private" if runtime == private / f"isaacsim-{VERSION}" else "external"}
    installation.save(record)
    if sdk is None:
        sdk = private / "sdk" / SDK_NAME
    private_sdk = private / "sdk" / SDK_NAME
    if not args.sdk_path and sdk in (private_sdk, local_sdk.resolve()):
        update_sdk.ensure_sdk(sdk, "linux-x86_64", managed=sdk == private_sdk,
                              state_dir=project / ".installation/sdk-updates")
        # Persist a completed replacement before later setup/load checks can fail.
        record_library(record, "sdk", sdk, private_sdk)
        installation.save(record)
    picker = prepare_picker(project, private, record)
    # Separate process catches ELF/ABI/dependent-library errors before starting Kit.
    subprocess.run([str(runtime_python(runtime)), "-B", "-c",
                    "import ctypes,sys; s=ctypes.CDLL(sys.argv[1]); "
                    "s.fs_create_context; s.fs_release_context; s.fs_find_surface; "
                    "p=ctypes.CDLL(sys.argv[2]); p.fsis_pick_perspective; p.fsis_pick_parallel",
                    str(sdk), str(picker)], check=True)
    record_library(record, "sdk", sdk, private / "sdk" / SDK_NAME)
    record_library(record, "picker", picker, private / "picker-build" / PICKER_NAME)
    installation.save(record)
    for key in ("runtime", "sdk", "picker"):
        print(f"{key} ({record[key]['ownership']}): {record[key]['path']}", flush=True)
    if args.prepare_only:
        return 0
    env = os.environ.copy()
    env["FSIS_PICKING_LIBRARY_PATH"] = str(picker)
    env["FINDSURFACE_SDK_AUTO_DOWNLOAD"] = "0"
    extra = args.kit_args[1:] if args.kit_args[:1] == ["--"] else args.kit_args
    return subprocess.run(repo.launch_command(repo.inspect_runtime(runtime),
                          [*extra, SETTING + str(sdk)]), cwd=runtime, env=env).returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "remove"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--install-base")
    parser.add_argument("--isaac-root", "--isaac-sim-root", dest="isaac_root")
    parser.add_argument("--sdk-path")
    # Parse launcher options before the explicit '--' separating Kit settings.
    values = list(sys.argv[1:] if argv is None else argv)
    split = values.index("--") if "--" in values else len(values)
    args = parser.parse_args(values[:split])
    args.kit_args = values[split:]
    try:
        if sys.version_info < (3, 10) or platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("Ubuntu x86_64 and Python 3.10+ are required; use Start_Demo.bat on Windows")
        if os.geteuid() == 0:
            raise ValueError("Run this launcher as a regular desktop user, not root/sudo")
        installation = Installation()
        with installation.lock():
            if args.action == "remove":
                if not installation.record_path.exists():
                    raise ValueError("No ownership record; nothing will be automatically removed")
                installation.remove(read_json(installation.record_path))
                return 0
            return start(args, installation)
    except (ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile, subprocess.CalledProcessError,
            repo.RuntimeDiscoveryError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
