"""Consent-based updates to explicitly approved, ABI-compatible SDK releases."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

POLICY_PATH = Path(__file__).with_name("sdk_releases.json")
MAX_ASSET_BYTES = 32 * 1024 * 1024
HEADERS = {"User-Agent": "FindSurfaceIsaacSimDemo-SDK-updater", "Accept": "application/vnd.github+json"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_path(path):
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and getattr(part.stat(), "st_file_attributes", 0)
                                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
            raise ValueError(f"Refusing a symlink/junction SDK path: {part}")


@contextmanager
def sdk_lock(path):
    with path.open("r+b" if path.exists() else "w+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def get_release(repository, tag):
    url = f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=10) as response:
        release = json.loads(response.read(2 * 1024 * 1024))
    if release.get("tag_name") != tag or release.get("draft") or release.get("prerelease"):
        raise ValueError("Release metadata does not describe the approved stable release")
    return release


def release_asset(repository, entry, key, metadata):
    expected = entry["assets"][key]
    url = f"https://github.com/{repository}/releases/download/{entry['tag']}/{expected['name']}"
    assets = [asset for asset in metadata.get("assets", []) if asset.get("name") == expected["name"]]
    if len(assets) != 1 or assets[0].get("browser_download_url") != url:
        raise ValueError("Approved SDK asset is missing or has an unexpected URL")
    published = str(assets[0].get("digest", ""))
    if published.startswith("sha256:"):
        published = published[7:]
    pinned = expected.get("sha256")
    if pinned and re.fullmatch(r"[0-9a-f]{64}", published) and pinned != published:
        raise ValueError("Published SDK digest differs from the approved binary")
    digest = pinned or published
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Release has no SHA-256 digest; keep the existing SDK until a digest is published")
    return url, digest


def download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": HEADERS["User-Agent"]})
    with urllib.request.urlopen(request, timeout=30) as response, destination.open("wb") as stream:
        if not response.geturl().startswith("https://"):
            raise ValueError("SDK download redirected away from HTTPS")
        received = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            received += len(block)
            if received > MAX_ASSET_BYTES:
                raise ValueError("SDK asset exceeds the download size limit")
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())


def validate_library(path):
    # A child process releases Windows DLL locks and contains native load failures.
    code = """import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
for name in ('fs_set_float', 'fs_set_int', 'fs_set_boolean', 'fs_get_boolean',
             'fs_set_pointcloud', 'fs_find_surface', 'fs_find_plane', 'fs_find_sphere',
             'fs_find_cylinder', 'fs_find_cone', 'fs_find_torus', 'fs_get_outlier_flags'):
    getattr(lib, name)
lib.fs_create_context.argtypes = [ctypes.c_bool]
lib.fs_create_context.restype = ctypes.c_void_p
lib.fs_release_context.argtypes = [ctypes.c_void_p]
lib.fs_release_context.restype = None
context = lib.fs_create_context(False)
if not context: raise RuntimeError('fs_create_context returned NULL')
lib.fs_release_context(context)
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(path)],
                            capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise ValueError("SDK load/ABI check failed: " + (result.stderr or str(result.returncode)).strip())


def installed_tag(path, policy, key, receipt):
    digest = sha256(path)
    for entry in policy["releases"]:
        if entry["assets"][key].get("sha256") == digest:
            return entry["tag"]
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record["sha256"] == digest and record["asset_key"] == key:
            if any(entry["tag"] == record["tag"] for entry in policy["releases"]):
                return record["tag"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def ask_update(old, new, path, metadata):
    print(f"FindSurface SDK update available: {old or 'app-managed SDK'} -> {new}")
    print(f"File to replace: {path}")
    print("The old binary will be replaced only after download, SHA-256 and ABI checks succeed.")
    print(f"Release: {metadata['html_url']}")
    notes = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", str(metadata.get("body") or ""))[:1200]
    if notes.strip():
        print(notes.strip())
    try:
        return input("Update now? [y/N]: ").strip().lower() in {"y", "yes"}
    except (EOFError, OSError):
        return False


def replace_sdk(path, entry, key, url, digest, old_digest, receipt):
    check_path(path)
    check_path(receipt)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".download", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        download(url, temporary)
        if sha256(temporary) != digest:
            raise ValueError("Downloaded SDK SHA-256 mismatch")
        validate_library(temporary)
        check_path(path)
        if (sha256(path) if path.exists() else None) != old_digest:
            raise ValueError("SDK changed during update; refusing to overwrite it")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    try:
        fd, name = tempfile.mkstemp(prefix="receipt-", suffix=".tmp", dir=receipt.parent)
        pending = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"tag": entry["tag"], "sha256": digest, "asset_key": key}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, receipt)
        finally:
            pending.unlink(missing_ok=True)
    except OSError as error:
        print(f"SDK installed, but its version receipt could not be saved: {error}")


def ensure_sdk(path, key, *, managed=False, policy=None, confirm=None, state_dir=None):
    policy = policy or json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    path = Path(path).absolute()
    check_path(path)
    state_dir = Path(state_dir) if state_dir else POLICY_PATH.parents[1] / ".installation" / "sdk-updates"
    check_path(state_dir)
    identity = hashlib.sha256(os.path.normcase(str(path)).encode("utf-8")).hexdigest()
    receipt = state_dir / (identity + ".json")
    current = installed_tag(path, policy, key, receipt) if path.exists() else None
    if path.exists() and not managed and current is None:
        print(f"Preserving unrecognized/custom SDK: {path}")
        return
    newest = policy["releases"][-1]
    if current == newest["tag"]:
        print(f"FindSurface SDK {current} is already installed.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        with sdk_lock(state_dir / (identity + ".lock")):
            old_digest = sha256(path) if path.exists() else None
            current = installed_tag(path, policy, key, receipt) if old_digest else None
            if old_digest and not managed and current is None:
                print(f"Preserving unrecognized/custom SDK: {path}")
                return
            if current == newest["tag"]:
                print(f"FindSurface SDK {current} is already installed.")
                return
            metadata = None
            try:
                print(f"Checking approved FindSurface SDK release {newest['tag']}...")
                metadata = get_release(policy["repository"], newest["tag"])
                url, digest = release_asset(policy["repository"], newest, key, metadata)
                selected = newest
            except urllib.error.HTTPError as error:
                if path.exists() or error.code != 404:
                    raise
                # The approved update may not have been published yet. Fresh installs
                # can still use the retained original release, with its pinned hash.
                selected = policy["releases"][0]
                asset = selected["assets"][key]
                url = f"https://github.com/{policy['repository']}/releases/download/{selected['tag']}/{asset['name']}"
                digest = asset["sha256"]
            if path.exists() and not (confirm or ask_update)(current, selected["tag"], path, metadata):
                print("SDK update declined; keeping the existing binary.")
                return
            replace_sdk(path, selected, key, url, digest, old_digest, receipt)
            print(f"FindSurface SDK {selected['tag']} installed: {path}")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        if not path.exists():
            raise
        print(f"SDK update unavailable/failed; keeping the existing binary: {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-path", required=True, type=Path)
    parser.add_argument("--managed", action="store_true", help="SDK is owned and verified by the private installer")
    parser.add_argument("--asset-key", choices=("windows-x86_64", "linux-x86_64"))
    args = parser.parse_args()
    key = args.asset_key or ("windows-x86_64" if sys.platform == "win32" else "linux-x86_64")
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        parser.error("Only x86-64 SDK binaries are supported")
    try:
        ensure_sdk(args.sdk_path, key, managed=args.managed)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
