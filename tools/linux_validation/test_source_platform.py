"""Regression for Linux source demos being filtered out as Windows-only."""
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest

from run import configure_test_app, prepare_test_core


class SourcePlatformTests(unittest.TestCase):
    def test_startup_popup_setting_is_idempotent_and_preserves_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            app = Path(directory) / "demo.kit"
            app.write_text('[settings]\napp.name="Demo"\n[dependencies]\n"demo"={}\n')
            configure_test_app(app)
            once = app.read_bytes()
            configure_test_app(app)
            self.assertEqual(app.read_bytes(), once)
            config = tomllib.loads(app.read_text())
            self.assertTrue(config["settings"]["persistent"]["renderer"]["startupMessageDisplayed"])
            self.assertIn("demo", config["dependencies"])

    def test_linux_candidate_changes_disposable_copy_only(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            original = base / "original"
            (original / "config").mkdir(parents=True)
            text = '[package]\nversion="1.1.0"\n[package.target]\nkit=["110.1.2"]\nplatform = ["windows-x86_64"]\n'
            (original / "config/extension.toml").write_text(text, encoding="utf-8")
            work = base / "work"
            shutil.copytree(original, work)
            sdk = base / "libFindSurface.so"
            sdk.write_bytes(b"SDK fixture")
            prepare_test_core(work, sdk, "linux-x86_64")
            actual = tomllib.loads((work / "config/extension.toml").read_text())
            self.assertEqual(actual["package"]["target"]["platform"], ["linux-x86_64"])
            self.assertEqual(actual["package"]["target"]["kit"], ["110.1.2"])
            self.assertEqual((original / "config/extension.toml").read_text(), text)
            self.assertEqual((work / "data/native/linux-x86_64/libFindSurface.so").read_bytes(), sdk.read_bytes())

    def test_missing_target_is_reported_instead_of_a_false_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            core = Path(directory)
            (core / "config").mkdir()
            (core / "config/extension.toml").write_text('[package]\nversion="1.1.0"\n')
            with self.assertRaisesRegex(ValueError, "exactly one"):
                prepare_test_core(core, core / "missing.so", "linux-x86_64")


if __name__ == "__main__":
    unittest.main()
