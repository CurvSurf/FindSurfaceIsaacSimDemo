"""Extract the pinned Linux ZIP with permissions, safe links and byte progress."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import stat
import time
import zipfile


def extract_archive(archive: Path, destination: Path) -> None:
    """Use a new staging directory; never follow archive-created links when writing."""
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Extraction destination already exists: {destination}")
    with zipfile.ZipFile(archive) as source:
        entries = []
        names = set()
        links = set()
        for entry in source.infolist():
            name = entry.filename.rstrip("/")
            path = PurePosixPath(name)
            mode = entry.external_attr >> 16
            canonical = path.as_posix()
            if (not path.parts or "\\" in name or path.is_absolute() or ".." in path.parts
                    or ":" in name or "\x00" in name or canonical in names):
                raise ValueError(f"Unsafe or duplicate ZIP path: {entry.filename!r}")
            kind = stat.S_IFMT(mode)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR, stat.S_IFLNK):
                raise ValueError(f"Unsupported ZIP entry: {name}")
            names.add(canonical)
            if stat.S_ISLNK(mode):
                links.add(path)
            entries.append((entry, path, mode))
        for _, path, _ in entries:
            if any(parent in links for parent in path.parents):
                raise ValueError(f"ZIP entry is below a symbolic link: {path}")
        # Validate link targets before creating even the staging directory.
        root = destination.absolute()
        targets = {}
        for entry, path, mode in entries:
            if not stat.S_ISLNK(mode):
                continue
            if entry.file_size > 4096:
                raise ValueError(f"Oversized ZIP link: {path}")
            target = source.read(entry).decode("utf-8")
            if not target or "\\" in target or PurePosixPath(target).is_absolute():
                raise ValueError(f"Unsafe ZIP link: {path}")
            resolved = Path(os.path.normpath(root / path.parent / target))
            if not resolved.is_relative_to(root):
                raise ValueError(f"ZIP link escapes extraction: {path}")
            targets[path] = target
        destination.mkdir(parents=True)
        total = sum(entry.file_size for entry, _, mode in entries if not stat.S_ISLNK(mode))
        done = 0
        started = last_report = time.monotonic()
        for index, (entry, path, mode) in enumerate(entries, 1):
            output = destination / path
            if stat.S_ISLNK(mode):
                continue
            if entry.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.open(entry) as reader, output.open("xb") as writer:
                while chunk := reader.read(1024 * 1024):
                    writer.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= 1:
                        elapsed = now - started
                        eta = (total - done) * elapsed / done if done else 0
                        print(f"Extracting {100 * done / max(1, total):5.1f}% | "
                              f"{done / 2**30:.2f}/{total / 2**30:.2f} GiB | "
                              f"file {index}/{len(entries)} | {elapsed:.0f}s elapsed | "
                              f"~{eta:.0f}s remaining", flush=True)
                        last_report = now
            output.chmod((mode & 0o777) or 0o644)
        # All regular files are in place before any symlink exists.
        for path, target in targets.items():
            output = destination / path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.symlink_to(target)
        for path in targets:
            try:
                resolved = (destination / path).resolve()
            except RuntimeError as error:
                raise ValueError(f"Cyclic ZIP link: {path}") from error
            if not resolved.is_relative_to(root):
                raise ValueError(f"ZIP link chain escapes extraction: {path}")
        # Directory execute bits must not be removed while children are extracted.
        for entry, path, mode in reversed(entries):
            if entry.is_dir():
                (destination / path).chmod((mode & 0o777) or 0o755)
        print(f"Extraction complete: {done / 2**30:.2f} GiB, "
              f"{time.monotonic() - started:.0f}s", flush=True)
