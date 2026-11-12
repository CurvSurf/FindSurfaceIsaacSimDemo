"""Read-only, bounded discovery of previously installed FindSurface runtimes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys


SKIP = {".git", ".artifacts", ".cache", ".local", ".venv", ".npm", ".cargo",
        "AppData", "node_modules", "__pycache__", "_build", "_repo", "_compiler",
        "build", "dist", "sources", "work", "installed", "native"}


def version(text):
    match = re.match(r"(\d+\.\d+\.\d+)(?:\D|$)", str(text).strip())
    return match.group(1) if match else ""


def validate_runtime(path, kind, platform_name):
    """Validate files and version before executing anything; preserve kit symlinks."""
    root = Path(os.path.abspath(Path(path).expanduser()))
    executable_name = "kit.exe" if platform_name.startswith("windows") else "kit"
    if root.is_file():
        root = root.parent
    if kind == "isaac":
        if root.name == "kit":
            root = root.parent
        executable = root / "kit" / executable_name
        actual = version((root / "VERSION").read_text(encoding="utf-8-sig"))
        expected = "6.0.1"
    else:
        if not (root / "kernel/config/kit-core.json").is_file() and (root / "kit").is_dir():
            root = root / "kit"
        executable = root / executable_name
        config = json.loads((root / "kernel/config/kit-core.json").read_text(encoding="utf-8-sig"))
        actual = version(config.get("crashreporter", {}).get("data", {}).get("buildVersion", ""))
        expected = "110.1.2"
    if actual != expected:
        raise ValueError(f"Expected {kind} {expected}, found {actual or 'unknown version'}")
    if not executable.is_file():
        raise ValueError(f"Executable missing: {executable}")
    if os.name != "nt" and not os.access(executable, os.X_OK):
        raise ValueError(f"Executable permission missing: {executable}")
    return root


def discover_runtimes(toolkit, kit_root=None, isaac_root=None, *, home=None, cwd=None,
                      environ=None, platform_name=None, max_depth=4):
    home = Path(home or Path.home()).absolute()
    cwd = Path(cwd or Path.cwd()).absolute()
    toolkit = Path(toolkit).absolute()
    environ = os.environ if environ is None else environ
    platform_name = platform_name or ("windows-x86_64" if sys.platform == "win32" else "linux-x86_64")
    report = {"selected": {}, "rejected": [], "search_roots": [], "checkout_paths": [], "errors": []}
    candidates = {"kit": [], "isaac": []}
    explicit = {"kit": kit_root, "isaac": isaac_root}

    def add(kind, path, reason):
        if path:
            candidates[kind].append((Path(path), reason))

    # Explicit input always wins, including failure: never silently replace a typo.
    for kind, supplied in explicit.items():
        if supplied is not None:
            try:
                root = validate_runtime(supplied, kind, platform_name)
                report["selected"][kind] = {"path": str(root), "source": "explicit option"}
            except (OSError, ValueError, TypeError) as error:
                report["errors"].append(f"Invalid --{kind}-root {supplied}: {error}")

    checkouts = []
    seen = set()
    roots = list(dict.fromkeys([cwd, *([toolkit.parent] if toolkit.parent.name not in SKIP else []), home]))
    budget = 3000
    for search_root in roots:
        report["search_roots"].append(str(search_root))
        pending = [(search_root, 0)]
        while pending and budget > 0:
            folder, depth = pending.pop(0)
            key = str(folder)
            if key in seen:
                continue
            seen.add(key)
            budget -= 1
            manifest = folder / ".installation/manifest.json"
            has_app = (folder / "source/apps/com.curvsurf.findsurface_omniverse_demo.kit").is_file() or (folder / "source/apps/com.curvsurf.findsurface_isaac_sim_demo.kit").is_file()
            if manifest.is_file() or has_app:
                checkouts.append(folder)
                if manifest.is_file():
                    try:
                        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
                        recorded = data.get("runtime", {}).get("path")
                        if recorded:
                            recorded = Path(recorded).expanduser()
                            if not recorded.is_absolute():
                                recorded = folder / recorded
                            # An Isaac installation's Kit must not displace a
                            # dedicated Kit installation recorded by the other demo.
                            kind = "isaac" if data.get("app") == "FindSurfaceIsaacSimDemo" or (recorded / "VERSION").is_file() else "kit"
                            add(kind, recorded, f"installation record: {manifest}")
                    except (OSError, ValueError, TypeError, AttributeError) as error:
                        report["rejected"].append({"path": str(manifest), "reason": str(error)})
                # Do not walk source files or installation data inside a found checkout.
                continue
            if depth < max_depth:
                try:
                    children = sorted(folder.iterdir(), key=lambda p: p.name.lower())
                    pending.extend((p, depth + 1) for p in children
                                   if not p.name.startswith(".") and p.name not in SKIP
                                   and p.is_dir() and not p.is_symlink())
                except OSError:
                    pass
    report["checkout_paths"] = [str(p) for p in checkouts]

    for variable in ("KIT_ROOT", "KIT_EXECUTABLE", "KIT_EXE"):
        add("kit", environ.get(variable), f"environment: {variable}")
    add("isaac", environ.get("ISAACSIM_ROOT"), "environment: ISAACSIM_ROOT")
    for checkout in checkouts:
        add("kit", checkout / "_build" / platform_name / "release/kit", f"checkout build: {checkout}")
        add("kit", checkout / "kit", f"checkout runtime: {checkout}")
    data_home = Path(environ.get("XDG_DATA_HOME", str(home / ".local/share"))).expanduser()
    for base in (data_home / "CurvSurf/FindSurfaceIsaacSimDemo",):
        try:
            for root in sorted(base.glob("fsis-*/isaacsim-6.0.1")):
                add("isaac", root, "FindSurface private installation")
        except OSError:
            pass
    for root in (home / "isaacsim-6.0.1", home / "isaacsim", Path("/opt/isaacsim")):
        add("isaac", root, "standard installation location")
    for root in (home / "kit-110.1.2", home / "kit"):
        add("kit", root, "standard installation location")
    if platform_name.startswith("windows"):
        add("isaac", Path("C:/isaacsim-6.0.1"), "standard Windows installation")

    def select(kind):
        if explicit[kind] is not None or kind in report["selected"]:
            return
        attempted = set()
        for path, reason in candidates[kind]:
            key = str(path)
            if key in attempted:
                continue
            attempted.add(key)
            try:
                root = validate_runtime(path, kind, platform_name)
                if reason == "compatible Kit inside discovered Isaac Sim":
                    loops = [*root.glob("exts/omni.kit.loop-default*/config/extension.toml"),
                             *root.glob("extscore/omni.kit.loop-default*/config/extension.toml"),
                             *root.parent.glob("extscache/omni.kit.loop-default*/config/extension.toml")]
                    if not loops:
                        raise ValueError("Isaac's embedded Kit lacks the standalone default loop; use a prepared Kit installation")
                report["selected"][kind] = {"path": str(root), "source": reason}
                return
            except (OSError, ValueError, TypeError) as error:
                if path.exists() and len(report["rejected"]) < 100:
                    report["rejected"].append({"kind": kind, "path": str(path), "source": reason, "reason": str(error)})

    select("isaac")
    if "isaac" in report["selected"]:
        add("kit", Path(report["selected"]["isaac"]["path"]) / "kit", "compatible Kit inside discovered Isaac Sim")
    select("kit")
    for kind in ("kit", "isaac"):
        if kind not in report["selected"] and explicit[kind] is None:
            report["errors"].append(f"Could not find a compatible {kind} runtime. Rerun with --{kind}-root /actual/installation/folder")
    report["ok"] = not report["errors"] and len(report["selected"]) == 2
    return report
