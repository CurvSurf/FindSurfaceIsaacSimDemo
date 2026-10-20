"""Host-only installation safety tests; no SDK/runtime downloads or GPU required."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import manage_installation as installer
from linux_archive import extract_archive


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.project = self.base / "app with spaces"
        self.project.mkdir()
        self.installation = installer.Installation(self.project)

    def tearDown(self):
        self.temp.cleanup()

    def test_owned_record_round_trip_and_external_removal(self):
        record = self.installation.create(self.base / "private dependencies")
        external = self.base / "external-sdk.so"
        external.write_bytes(b"external")
        record["sdk"] = {"path": str(external), "ownership": "external"}
        self.installation.save(record)
        private = self.installation.owned_root(record)
        (private / "download.zip").write_bytes(b"owned")
        if not shutil.rmtree.avoids_symlink_attacks:
            self.skipTest("fd-safe removal is a Linux requirement")
        self.installation.remove(record, confirm=lambda _: "REMOVE")
        self.assertFalse(private.exists())
        self.assertTrue(external.exists())
        self.assertTrue(self.project.exists())
        self.assertFalse(self.installation.record_path.exists())

    def test_moved_checkout_or_changed_owner_cannot_remove(self):
        record = self.installation.create(self.base / "private")
        other = installer.Installation(self.base)
        with self.assertRaisesRegex(ValueError, "identity"):
            other.owned_root(record)
        root = Path(record["private_root"])
        installer.write_json(root / ".fsis-owner.json", {"app": "wrong"})
        with self.assertRaisesRegex(ValueError, "marker"):
            self.installation.owned_root(record)

    def test_removal_cancelled_preserves_everything(self):
        record = self.installation.create(self.base / "private")
        self.installation.remove(record, confirm=lambda _: "no")
        self.assertTrue(Path(record["private_root"]).exists())
        self.assertTrue(self.installation.record_path.exists())

    def test_private_library_tampering_rejected(self):
        path = self.base / "sdk.so"
        path.write_bytes(b"original")
        record = {}
        installer.record_library(record, "sdk", path, path)
        self.assertEqual(installer.recorded_library(record, "sdk", path), path)
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            installer.recorded_library(record, "sdk", path)
        with self.assertRaisesRegex(ValueError, "path mismatch"):
            installer.recorded_library(record, "sdk", self.base / "elsewhere.so")

    def test_approved_replacement_recovers_stale_private_sdk_record(self):
        path = self.base / "sdk.so"
        path.write_bytes(b"old")
        record = {}
        installer.record_library(record, "sdk", path, path)
        path.write_bytes(b"approved replacement")
        policy = {"releases": [{"assets": {"linux-x86_64": {"sha256": installer.checksum(path)}}}]}
        with patch.object(installer, "read_json", return_value=policy):
            self.assertEqual(installer.recorded_library(record, "sdk", path), path)

    def test_verified_cached_download_no_network_and_corruption_retained(self):
        path = self.base / "archive.zip"
        path.write_bytes(b"valid")
        digest = hashlib.sha256(b"valid").hexdigest()
        with patch.object(installer.subprocess, "run") as run:
            installer.receive("https://example.invalid/archive.zip", path, digest)
            run.assert_not_called()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            installer.receive("https://example.invalid/archive.zip", path, "0" * 64)
        self.assertEqual(path.read_bytes(), b"valid")
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            installer.receive("http://example.invalid/archive.zip", path, digest)

    @unittest.skipUnless(sys.platform == "linux", "Linux filesystem/process semantics")
    def test_symlink_owner_path_rejected_and_child_link_target_preserved(self):
        external = self.base / "external"
        external.mkdir()
        (external / "keep").write_text("safe")
        link = self.base / "link"
        link.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symbolic link"):
            self.installation.create(link)
        record = self.installation.create(self.base / "private")
        (Path(record["private_root"]) / "child-link").symlink_to(external, target_is_directory=True)
        self.installation.remove(record, confirm=lambda _: "REMOVE")
        self.assertEqual((external / "keep").read_text(), "safe")

    @unittest.skipUnless(sys.platform == "linux", "Linux flock")
    def test_checkout_lock_excludes_second_operation(self):
        with self.installation.lock():
            with self.assertRaisesRegex(ValueError, "already running"):
                with self.installation.lock():
                    self.fail("second operation acquired lock")

    def test_sdk_release_policy_has_approved_files_for_both_platforms(self):
        policy = installer.read_json(installer.update_sdk.POLICY_PATH)
        self.assertEqual(policy["repository"], "CurvSurf/FindSurfaceOmniverseExtension")
        latest = policy["releases"][-1]
        self.assertEqual(latest["tag"], "v1.0.1")
        for key, name in (("windows-x86_64", "FindSurface.dll"), ("linux-x86_64", "libFindSurface.so")):
            self.assertEqual(latest["assets"][key]["name"], name)
            self.assertRegex(latest["assets"][key]["sha256"], r"^[0-9a-f]{64}$")

    def test_valid_and_explicit_invalid_runtime(self):
        runtime = self.base / "runtime"
        (runtime / "kit/python/bin").mkdir(parents=True)
        (runtime / "apps").mkdir()
        (runtime / "VERSION").write_text("6.0.1+test")
        for file in (runtime / "kit/kit", runtime / "kit/python/bin/python3", installer.runtime_python(runtime)):
            file.write_text("test")
            file.chmod(0o755)
        self.assertTrue(installer.valid_runtime(runtime))
        self.assertEqual(installer.find_runtime(str(runtime), {}, self.base), runtime)
        (runtime / "VERSION").write_text("5.1.0")
        with self.assertRaisesRegex(ValueError, "Explicit runtime"):
            installer.find_runtime(str(runtime), {}, self.base)

    @unittest.skipUnless(sys.platform == "linux", "Linux ELF build")
    def test_picker_build_reuse_and_source_change_invalidation(self):
        if not shutil.which("cmake") or not shutil.which("c++"):
            self.skipTest("native build tools missing")
        source = self.project / "source/native"
        shutil.copytree(ROOT / "source/native", source)
        private = self.base / "build root"
        record = {}
        built = installer.prepare_picker(self.project, private, record)
        import ctypes
        library = ctypes.CDLL(str(built))
        class Result(ctypes.Structure):
            _fields_ = [("index", ctypes.c_int32), ("depth", ctypes.c_double),
                        ("distance_sq", ctypes.c_double), ("fallback", ctypes.c_uint8)]
        points = (ctypes.c_float * 9)(0, 0, 2, 1, 0, 3, 0, 0, -1)
        origin = (ctypes.c_double * 3)(0, 0, 0)
        direction = (ctypes.c_double * 3)(0, 0, 1)
        for name in ("fsis_pick_perspective", "fsis_pick_parallel"):
            pick = getattr(library, name)
            pick.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double,
                             ctypes.c_int32, ctypes.POINTER(Result)]
            pick.restype = ctypes.c_int
            result = Result()
            self.assertEqual(pick(points, None, 3, origin, direction, 0.1, 0, ctypes.byref(result)), 1)
            self.assertEqual(result.index, 0)
            self.assertEqual(result.depth, 2)
            mask = (ctypes.c_uint8 * 3)(0, 1, 1)
            self.assertEqual(pick(points, mask, 3, origin, direction, 0.1, 0, ctypes.byref(result)), 0)
            self.assertEqual(pick(points, mask, 3, origin, direction, 0.1, 1, ctypes.byref(result)), 1)
            self.assertEqual(result.index, 1)
            self.assertEqual(result.fallback, 1)
        installer.record_library(record, "picker", built, built)
        with patch.object(installer.subprocess, "run") as run:
            self.assertEqual(installer.prepare_picker(self.project, private, record), built)
            run.assert_not_called()
        with (source / "FindSurfacePicking.cpp").open("a") as stream:
            stream.write("\n// Source update test\n")
        with patch.object(installer.subprocess, "run") as run:
            installer.prepare_picker(self.project, private, record)
            self.assertEqual(run.call_count, 2)

    def fake_dependencies(self):
        """Tiny executable fixtures exercise setup/launch, not Isaac Sim or the SDK."""
        runtime = self.base / "external runtime"
        (runtime / "kit/python/bin").mkdir(parents=True)
        (runtime / "apps").mkdir()
        (runtime / "VERSION").write_text("6.0.1+fixture")
        python = runtime / "kit/python/bin/python3"
        python.symlink_to(sys.executable)
        wrapper = installer.runtime_python(runtime)
        wrapper.write_text('#!/bin/sh\nexec "' + sys.executable + '" "$@"\n')
        wrapper.chmod(0o755)
        kit = runtime / "kit/kit"
        kit.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                       'from pathlib import Path\n'
                       'Path(os.environ["FSIS_TEST_LOG"]).write_text(json.dumps({\n'
                       '"args":sys.argv[1:],"picker":os.environ["FSIS_PICKING_LIBRARY_PATH"],\n'
                       '"download":os.environ["FINDSURFACE_SDK_AUTO_DOWNLOAD"]}))\n')
        kit.chmod(0o755)
        bootstrap = self.project / "source/extensions/com.findsurface.bootstrap/config"
        bootstrap.mkdir(parents=True)
        (bootstrap / "extension.toml").write_text("[package]\n")
        (self.project / "tools").mkdir()
        installer.write_json(self.project / "tools/installation_dependencies.json", {
            "linux-x86_64": {"isaac_url": "https://example.invalid/runtime.zip",
                             "isaac_md5": "0" * 32}})
        shutil.copytree(ROOT / "source/native", self.project / "source/native")
        sdk = self.base / "external SDK.so"
        subprocess.run(["c++", "-shared", "-fPIC", "-x", "c++", "-", "-o", str(sdk)],
                       input=b'extern "C" { void fs_create_context(){} void fs_release_context(){} '
                             b'void fs_find_surface(){} }', check=True)
        args = type("Args", (), {"sdk_path": str(sdk), "isaac_root": str(runtime),
                                "install_base": str(self.base / "private"),
                                "prepare_only": False, "kit_args": ["--", "--/test/value=with spaces"]})()
        return runtime, sdk, args

    @unittest.skipUnless(sys.platform == "linux", "Linux executable fixtures")
    def test_start_reuse_launch_arguments_and_remove_external_dependencies(self):
        runtime, sdk, args = self.fake_dependencies()
        log = self.base / "launch.json"
        with patch.dict(os.environ, {"FSIS_TEST_LOG": str(log)}):
            self.assertEqual(installer.start(args, self.installation), 0)
        record = installer.read_json(self.installation.record_path)
        self.assertEqual(record["runtime"]["ownership"], "external")
        self.assertEqual(record["sdk"]["ownership"], "external")
        self.assertEqual(record["picker"]["ownership"], "private")
        launched = json.loads(log.read_text())
        self.assertIn(installer.SETTING + str(sdk), launched["args"])
        self.assertIn("--/test/value=with spaces", launched["args"])
        self.assertEqual(launched["picker"], record["picker"]["path"])
        self.assertEqual(launched["download"], "0")
        args.prepare_only = True
        with patch.object(installer, "receive") as receive:
            self.assertEqual(installer.start(args, self.installation), 0)
            receive.assert_not_called()
        self.installation.remove(record, confirm=lambda _: "REMOVE")
        self.assertTrue(runtime.is_dir())
        self.assertTrue(sdk.is_file())

    @unittest.skipUnless(sys.platform == "linux", "Linux executable fixtures")
    def test_start_prepares_private_sdk_and_reuses_it_on_next_launch(self):
        runtime, fixture, args = self.fake_dependencies()
        args.sdk_path = None
        args.prepare_only = True
        def prepare(path, key, **options):
            self.assertEqual(key, "linux-x86_64")
            self.assertTrue(options["managed"])
            self.assertEqual(options["state_dir"], self.project / ".installation/sdk-updates")
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixture, path)
        with patch.object(installer.update_sdk, "ensure_sdk", side_effect=prepare) as update:
            self.assertEqual(installer.start(args, self.installation), 0)
            record = installer.read_json(self.installation.record_path)
            self.assertEqual(record["sdk"]["ownership"], "private")
            self.assertEqual(installer.start(args, self.installation), 0)
            self.assertEqual(update.call_count, 2)
            self.assertEqual(update.call_args.args[0], Path(record["sdk"]["path"]))

    @unittest.skipUnless(sys.platform == "linux", "Linux executable fixtures")
    def test_checkout_sdk_is_checked_but_explicit_sdk_is_never_updated(self):
        _, fixture, args = self.fake_dependencies()
        args.prepare_only = True
        with patch.object(installer.update_sdk, "ensure_sdk") as update:
            self.assertEqual(installer.start(args, self.installation), 0)
            update.assert_not_called()
        local = self.project / "source/extensions/com.findsurface.bootstrap/data/native/linux-x86_64" / installer.SDK_NAME
        local.parent.mkdir(parents=True)
        shutil.copyfile(fixture, local)
        # Simulate a checkout with no previous SDK selection.
        record = installer.read_json(self.installation.record_path)
        del record["sdk"]
        self.installation.save(record)
        args.sdk_path = None
        with patch.object(installer.update_sdk, "ensure_sdk") as update:
            self.assertEqual(installer.start(args, self.installation), 0)
            update.assert_called_once_with(local, "linux-x86_64", managed=False,
                                          state_dir=self.project / ".installation/sdk-updates")

    @unittest.skipUnless(sys.platform == "linux", "Linux extraction/install fixtures")
    def test_private_runtime_install_and_safe_removal(self):
        runtime, sdk, args = self.fake_dependencies()
        archive = self.base / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            for path in runtime.rglob("*"):
                zipped.write(path, path.relative_to(runtime).as_posix())
        args.isaac_root = None
        args.prepare_only = True
        def receive_fixture(url, destination, expected, algorithm="sha256"):
            self.assertEqual(algorithm, "md5")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive, destination)
        with patch.object(installer, "find_runtime", return_value=None), \
                patch.object(installer, "receive", side_effect=receive_fixture), \
                patch("builtins.input", return_value="YES"), \
                patch.object(installer.shutil, "disk_usage", return_value=type("Disk", (), {"free": 100 * 2**30})()):
            self.assertEqual(installer.start(args, self.installation), 0)
        record = installer.read_json(self.installation.record_path)
        self.assertEqual(record["runtime"]["ownership"], "private")
        installed = Path(record["runtime"]["path"])
        self.assertTrue(installer.valid_runtime(installed))
        self.installation.remove(record, confirm=lambda _: "REMOVE")
        self.assertFalse(installed.exists())
        self.assertTrue(sdk.exists())


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.archive = self.base / "runtime.zip"
        self.destination = self.base / "staging"

    def tearDown(self):
        self.temp.cleanup()

    def make_zip(self, files):
        with zipfile.ZipFile(self.archive, "w") as archive:
            for name, content, mode in files:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = mode << 16
                archive.writestr(info, content)

    def test_extract_unicode_and_refuse_existing_destination(self):
        self.make_zip([("folder/문자.txt", "hello", stat.S_IFREG | 0o644)])
        extract_archive(self.archive, self.destination)
        self.assertEqual((self.destination / "folder/문자.txt").read_text(), "hello")
        with self.assertRaisesRegex(ValueError, "already exists"):
            extract_archive(self.archive, self.destination)

    def test_traversal_and_duplicate_and_special_file_rejected_before_writing(self):
        cases = [ [("../escape", "bad", stat.S_IFREG | 0o644)],
                  [("/absolute", "bad", stat.S_IFREG | 0o644)],
                  [("C:/drive", "bad", stat.S_IFREG | 0o644)],
                  [("a//b", "one", stat.S_IFREG), ("a/b", "two", stat.S_IFREG)],
                  [("device", "bad", stat.S_IFCHR | 0o600)],
                  [("link", "../escape", stat.S_IFLNK | 0o777)],
                  [("link", "target", stat.S_IFLNK), ("link/child", "bad", stat.S_IFREG)] ]
        for files in cases:
            with self.subTest(files=files):
                self.make_zip(files)
                with self.assertRaises(ValueError):
                    extract_archive(self.archive, self.destination)
                self.assertFalse(self.destination.exists())

    @unittest.skipUnless(sys.platform == "linux", "Linux symlinks and executable modes")
    def test_internal_link_and_executable_bits_preserved(self):
        self.make_zip([("kit/kit", "test", stat.S_IFREG | 0o755),
                       ("kit/current", "kit", stat.S_IFLNK | 0o777)])
        extract_archive(self.archive, self.destination)
        self.assertTrue(os.access(self.destination / "kit/kit", os.X_OK))
        self.assertEqual((self.destination / "kit/current").resolve(), self.destination / "kit/kit")


if __name__ == "__main__":
    unittest.main()
