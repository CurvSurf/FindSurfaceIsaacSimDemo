"""Portable, unattended validation. Run in a logged-in Linux desktop session."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import traceback
import zipfile

from runtime_discovery import discover_runtimes

STARTUP_DIALOG_OPTION = "--/persistent/renderer/startupMessageDisplayed=true"


def configure_test_app(app):
    """Mirror the demo display setting in disposable committed-source copies."""
    text = app.read_text(encoding="utf-8")
    setting = "persistent.renderer.startupMessageDisplayed = true"
    pattern = r'^persistent\.renderer\.startupMessageDisplayed\s*=.*$'
    if re.search(pattern, text, flags=re.M):
        text = re.sub(pattern, setting, text, flags=re.M)
    elif "[settings]\n" in text:
        text = text.replace("[settings]\n", "[settings]\n" + setting + "\n", 1)
    else:
        raise ValueError(f"Missing app settings section: {app}")
    app.write_text(text, encoding="utf-8", newline="\n")


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_test_core(core, sdk, platform_name):
    """Adjust only the disposable source copy, including direct-run targets."""
    config = core / "config/extension.toml"
    text, count = re.subn(r'^platform\s*=\s*\[.*\]$', f'platform = ["{platform_name}"]',
                          config.read_text(encoding="utf-8"), flags=re.M)
    if count != 1:
        raise ValueError(f"Expected exactly one core platform target in {config}")
    config.write_text(text, encoding="utf-8", newline="\n")
    dest = core / "data/native" / platform_name / sdk.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sdk, dest)


def run_command(command, log, timeout=300, cwd=None):
    """Keep all logs and stop only this test's process tree on timeout."""
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([str(arg) for arg in command], cwd=cwd, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=stream, stderr=subprocess.STDOUT)
            else:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise


def require_success(command, log, **kwargs):
    code = run_command(command, log, **kwargs)
    if code:
        raise RuntimeError(f"Exit {code}; see {log.name}")


def host_check(case, phase, root, installed, runtime, folder, smoke):
    launcher = {"demo": "com.curvsurf.findsurface_omniverse_launcher", "isaac": "com.curvsurf.findsurface_isaac_sim_launcher"}.get(case)
    main = {"demo": "com.curvsurf.findsurface_omniverse", "isaac": "com.curvsurf.findsurface_isaac_sim"}.get(case)
    module = {"demo": "findsurface_omniverse", "isaac": "findsurface_isaac_sim"}.get(case)
    executable = runtime / (("kit/kit.exe" if os.name == "nt" else "kit/kit") if case == "isaac" else ("kit.exe" if os.name == "nt" else "kit"))
    config = dict(case=case, phase=phase, result=str(folder / f"{phase}.json"), runtime=str(runtime),
                  launcher=launcher, main=main, module=module, smoke_script=str(smoke))
    if phase == "host":
        child = dict(config, phase="child", result=str(folder / "child.json"))
        write(folder / "child-config.json", child)
        config.update(launcher_root=str(installed / case / "exts" / launcher), child_config=str(folder / "child-config.json"),
                      child_result=child["result"], child_log=str(folder / "child.log"))
    extension = "com.findsurface.bootstrap" if case == "core" else launcher
    config["extension"] = extension
    write(folder / f"{phase}-config.json", config)
    app = folder / f"{phase}.kit"
    # Load USD/core during normal startup, before app-ready. Enabling the core
    # only in --exec caused spurious late omni.usd app-ready errors on Linux.
    dependencies = ('"omni.kit.uiapp" = {}\n"omni.kit.registry.nucleus" = {}\n'
                    '"com.findsurface.bootstrap" = { version = "1.1.0", exact = true }\n')
    if phase == "host":
        dependencies += f'"{launcher}" = {{}}\n'
    app.write_text('[dependencies]\n' + dependencies + '[settings]\napp.extensions.registryEnabled=false\napp.window.width=760\napp.window.height=440\n', encoding="utf-8")
    command = [executable, app, "--ext-folder", installed / "core/exts"]
    if case != "core":
        command += ["--ext-folder", installed / case / "exts"]
    for extra in ([runtime / "exts", runtime / "extscache", runtime / "apps"] if case == "isaac" else [runtime.parent / "extscache"]):
        if extra.is_dir():
            command += ["--ext-folder", extra]
    command += ["--exec", smoke, "--/findsurface/validation/config=" + str(folder / f"{phase}-config.json"),
                "--/log/file=" + str(folder / f"{phase}.log"), "--/app/settings/persistent=0", "--/app/settings/loadUserConfig=0", STARTUP_DIALOG_OPTION]
    require_success(command, folder / f"{phase}-console.log", timeout=240, cwd=runtime)
    result = json.loads(Path(config["result"]).read_text())
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "Kit check failed"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit-root", type=Path, help="override automatic discovery of Kit 110.1.2")
    parser.add_argument("--isaac-root", type=Path, help="override automatic discovery of Isaac Sim 6.0.1")
    parser.add_argument("--platform", choices=["linux-x86_64", "windows-x86_64"], default="linux-x86_64")
    parser.add_argument("--output", type=Path, help="fresh results folder")
    parser.add_argument("--skip-native-build", action="store_true", help="use already prepared picker libraries")
    args = parser.parse_args(argv)
    toolkit = Path(__file__).resolve().parent
    sources = toolkit / "sources"
    actual = "windows-x86_64" if sys.platform == "win32" else "linux-x86_64"
    if args.platform != actual or platform.machine().lower() not in {"amd64", "x86_64"}:
        parser.error("Run on the selected operating system and x86-64 hardware")
    output = (args.output or toolkit / ("results-" + datetime.now().strftime("%Y%m%d-%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"published": False, "platform": args.platform, "started_utc": datetime.now(timezone.utc).isoformat(),
              "system": platform.platform(), "python": sys.version, "steps": {}, "coverage": ["metadata", "native loading", "launcher lifecycle", "Isaac scene startup", "source and packaged app startup"],
              "manual_followup": ["visual rendering quality", "interactive picking and detection", "DepthSensor/LiDAR scan quality"]}
    write(output / "summary.json", report)
    work = output / "work"
    if not sources.is_dir():
        report["error"] = "sources folder missing; use the complete transfer kit"
        write(output / "summary.json", report)
        return 1
    shutil.copytree(sources, work)
    for case, app in [("demo", "com.curvsurf.findsurface_omniverse_demo.kit"),
                      ("isaac", "com.curvsurf.findsurface_isaac_sim_demo.kit")]:
        configure_test_app(work / case / "source/apps" / app)
    report["test_overlays"] = {"renderer_startup_popup": "hidden; warnings retained in logs"}
    installed = output / "installed"
    native_inputs = toolkit / "native" / args.platform

    def step(name, function):
        print(f"[{name}] starting", flush=True)
        try:
            result = function()
            report["steps"][name] = {"ok": True, "details": result}
            print(f"[{name}] PASS", flush=True)
            return True
        except Exception:
            report["steps"][name] = {"ok": False, "error": traceback.format_exc()}
            print(f"[{name}] FAIL (see summary.json)", flush=True)
            return False
        finally:
            write(output / "summary.json", report)

    def preflight():
        discovery = discover_runtimes(toolkit, args.kit_root, args.isaac_root, platform_name=args.platform)
        write(output / "runtime-discovery.json", discovery)
        for kind, selected in discovery["selected"].items():
            print(f"  {kind}: {selected['path']} ({selected['source']})", flush=True)
            setattr(args, kind + "_root", Path(selected["path"]))
        if not discovery["ok"]:
            for error in discovery["errors"]:
                print("  " + error, flush=True)
            raise RuntimeError("\n".join(discovery["errors"]))
        if args.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("Run from a logged-in graphical desktop session; this OpenGL test is not headless")
        if shutil.which("nvidia-smi"):
            require_success(["nvidia-smi"], output / "nvidia-smi.txt", timeout=30)
        if not args.skip_native_build and not shutil.which("cmake"):
            raise RuntimeError("Install CMake and a C++ compiler before running; the runner never uses sudo")
        for runtime, executable in [(args.kit_root, "kit.exe" if os.name == "nt" else "kit"), (args.isaac_root, "kit/kit.exe" if os.name == "nt" else "kit/kit")]:
            if not (runtime.expanduser() / executable).is_file():
                raise RuntimeError(f"Runtime executable missing: {runtime / executable}")
        return {"kit": str(args.kit_root), "isaac": str(args.isaac_root)}

    ready = step("environment", preflight)
    args.kit_root = (args.kit_root or toolkit / "MISSING_KIT_RUNTIME").expanduser().absolute()
    args.isaac_root = (args.isaac_root or toolkit / "MISSING_ISAAC_RUNTIME").expanduser().absolute()
    sdk_name = "FindSurface.dll" if os.name == "nt" else "libFindSurface.so"

    def sdk():
        path = native_inputs / sdk_name
        policy = json.loads((work / "core/config/sdk_release.json").read_text())
        expected = policy["sha256"][args.platform]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Supply pinned SDK {policy['release_tag']} in {path}")
        for core in [work / "core", work / "demo/source/extensions/com.findsurface.bootstrap", work / "isaac/source/extensions/com.findsurface.bootstrap"]:
            prepare_test_core(core, path, args.platform)
        return {"release": policy["release_tag"], "sha256": expected,
                "disposable_source_target": args.platform}

    sdk_ready = step("sdk", sdk)
    step("unit-tests", lambda: require_success([sys.executable, "-m", "unittest", "discover", "-s", work / "core/tests", "-q"], output / "unit-tests.log"))
    smoke = toolkit / "kit_smoke.py"
    package_tool = work / "core/tools/prepare_community_release.py"
    packages = {}

    for case in ["core", "demo", "isaac"]:
        case_folder = output / case
        case_folder.mkdir()

        def package(case=case, folder=case_folder):
            if not sdk_ready:
                raise RuntimeError("SDK preflight failed")
            project = work / case
            # Restrict the disposable test release to the platform under validation.
            config_path = project / "community_release.json"
            config = json.loads(config_path.read_text())
            config["supported_platforms"] = [args.platform]
            write(config_path, config)
            if case != "core" and not args.skip_native_build:
                if not ready:
                    raise RuntimeError("Environment preflight failed")
                build = output / "build" / case
                dest = project / "source/extensions" / ("com.curvsurf.findsurface_omniverse" if case == "demo" else "com.curvsurf.findsurface_isaac_sim") / "data/native" / args.platform
                dest.mkdir(parents=True, exist_ok=True)
                require_success(["cmake", "-S", project / "source/native", "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DFS_PICKER_OUTPUT_DIR=" + str(dest), "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(dest)], folder / "configure.log")
                require_success(["cmake", "--build", build, "--config", "Release", "--parallel", "2"], folder / "build.log")
            release = folder / "release"
            require_success([sys.executable, package_tool, "--project-root", project, "--output-dir", release, "--platform", args.platform], folder / "package.log")
            manifest = json.loads((release / "release-manifest.json").read_text())
            packages[case] = manifest
            target = installed / case
            target.mkdir(parents=True)
            # The builder checked member names and hashes before extraction.
            with zipfile.ZipFile(release / manifest["archive"]) as zipped:
                zipped.extractall(target)
            return {"archive": manifest["archive"], "sha256": manifest["archive_sha256"]}

        if step(case + "-package", package) and ready:
            runtime = args.isaac_root if case == "isaac" else args.kit_root
            step(case + "-metadata", lambda c=case, r=runtime, f=case_folder: host_check(c, "verify", work, installed, r, f, smoke))
            if case != "core":
                step(case + "-launcher", lambda c=case, r=runtime, f=case_folder: host_check(c, "host", work, installed, r, f, smoke))
                # The same demo experience is also exercised directly, without a launcher.
                def direct(c=case, r=runtime, f=case_folder):
                    spec = json.loads((work / c / "source/extensions" / packages[c]["extension"] / "config/demo.json").read_text())
                    executable = r / (("kit/kit.exe" if os.name == "nt" else "kit/kit") if c == "isaac" else ("kit.exe" if os.name == "nt" else "kit"))
                    cfg = json.loads((f / "child-config.json").read_text())
                    cfg["result"] = str(f / "direct.json")
                    write(f / "direct-config.json", cfg)
                    command = [executable, work / c / "source/apps" / spec["experience"], "--ext-folder", work / c / "source/extensions"]
                    for extra in ([r / "exts", r / "extscache", r / "apps"] if c == "isaac" else [r.parent / "extscache"]):
                        if extra.is_dir(): command += ["--ext-folder", extra]
                    command += ["--exec", smoke, "--/findsurface/validation/config=" + str(f / "direct-config.json"), "--/log/file=" + str(f / "direct.log"), "--/app/extensions/registryEnabled=0", "--/app/settings/persistent=0", "--/app/settings/loadUserConfig=0", STARTUP_DIALOG_OPTION]
                    require_success(command, f / "direct-console.log", timeout=240, cwd=r)
                    result = json.loads((f / "direct.json").read_text())
                    if not result.get("ok"): raise RuntimeError(result)
                    return result
                step(case + "-direct", direct)
    report["ok"] = all(row["ok"] for row in report["steps"].values())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write(output / "summary.json", report)
    bundle = output.with_suffix(".results.zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.relative_to(output).parts[0] not in {"work", "installed", "build"}:
                zipped.write(path, path.relative_to(output).as_posix())
    print(f"{'PASS' if report['ok'] else 'FAIL'}: {output / 'summary.json'}\nResult bundle: {bundle}", flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
