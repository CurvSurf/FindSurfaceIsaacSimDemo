"""Discovery tests use fake installation metadata; no Kit executable is run."""
import json
from pathlib import Path
import tempfile
import unittest

from runtime_discovery import discover_runtimes


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.toolkit = self.home / "validation"
        self.toolkit.mkdir()

    def kit(self, root, version="110.1.2"):
        config = root / "kernel/config/kit-core.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({"crashreporter": {"data": {"buildVersion": version}}}))
        binary = root / "kit"
        binary.touch()
        binary.chmod(0o755)
        loop = root / "extscore/omni.kit.loop-default/config/extension.toml"
        loop.parent.mkdir(parents=True, exist_ok=True)
        loop.touch()
        return root

    def isaac(self, root, version="6.0.1-rc.7+release.test", kit_version="110.1.2"):
        self.kit(root / "kit", kit_version)
        (root / "VERSION").write_text(version)
        return root

    def manifest(self, checkout, runtime):
        f = checkout / ".installation/manifest.json"
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps({"runtime": {"path": str(runtime)}}))

    def discover(self, **kwargs):
        return discover_runtimes(self.toolkit, home=self.home, cwd=self.toolkit,
                                 environ=kwargs.pop("environ", {}), platform_name="linux-x86_64", **kwargs)

    def test_reads_record_from_renamed_checkout(self):
        runtime = self.isaac(self.home / ".private/runtime")
        checkout = self.home / "projects/renamed-checkout"
        self.manifest(checkout, runtime)
        result = self.discover()
        self.assertTrue(result["ok"], result)
        self.assertEqual(Path(result["selected"]["isaac"]["path"]), runtime)
        self.assertIn("installation record", result["selected"]["isaac"]["source"])
        self.assertEqual(Path(result["selected"]["kit"]["path"]), runtime / "kit")

    def test_checkout_build_precedes_shared_isaac_kit(self):
        isaac = self.isaac(self.home / "isaacsim-6.0.1")
        checkout = self.home / "dev/arbitrary-name"
        app = checkout / "source/apps/com.curvsurf.findsurface_omniverse_demo.kit"
        app.parent.mkdir(parents=True)
        app.touch()
        kit = self.kit(checkout / "_build/linux-x86_64/release/kit")
        result = self.discover()
        self.assertTrue(result["ok"], result)
        self.assertEqual(Path(result["selected"]["kit"]["path"]), kit)

    def test_private_installation_respects_xdg_data_home(self):
        data = self.home / "custom-data"
        runtime = self.isaac(data / "CurvSurf/FindSurfaceIsaacSimDemo/fsis-id/isaacsim-6.0.1")
        result = self.discover(environ={"XDG_DATA_HOME": str(data)})
        self.assertTrue(result["ok"], result)
        self.assertEqual(Path(result["selected"]["isaac"]["path"]), runtime)

    def test_broken_record_does_not_block_valid_installation(self):
        f = self.home / "projects/old/.installation/manifest.json"
        f.parent.mkdir(parents=True)
        f.write_text("not json")
        self.isaac(self.home / "isaacsim-6.0.1")
        result = self.discover()
        self.assertTrue(result["ok"], result)
        self.assertTrue(result["rejected"])

    def test_invalid_explicit_override_is_not_silently_replaced(self):
        self.isaac(self.home / "isaacsim-6.0.1")
        result = self.discover(isaac_root=self.home / "typo")
        self.assertFalse(result["ok"])
        self.assertNotIn("isaac", result["selected"])
        self.assertIn("--isaac-root", result["errors"][0])

    def test_wrong_version_rejected_before_execution(self):
        self.isaac(self.home / "isaacsim-6.0.1", version="6.1.0")
        result = self.discover()
        self.assertFalse(result["ok"])
        self.assertTrue(any("6.1.0" in entry["reason"] for entry in result["rejected"]))

    def test_missing_runtime_reports_override_options(self):
        result = self.discover()
        self.assertFalse(result["ok"])
        self.assertTrue(any("--kit-root" in text for text in result["errors"]))
        self.assertTrue(any("--isaac-root" in text for text in result["errors"]))

    def test_explicit_paths_override_discovered_installations(self):
        self.isaac(self.home / "isaacsim-6.0.1")
        chosen = self.isaac(self.home / "elsewhere/chosen")
        result = self.discover(isaac_root=chosen, kit_root=chosen / "kit")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["selected"]["kit"]["source"], "explicit option")

    def test_dedicated_kit_record_precedes_embedded_isaac_kit(self):
        isaac = self.isaac(self.home / ".private/isaac")
        kit = self.kit(self.home / ".private/kit")
        self.manifest(self.home / "projects/a-isaac", isaac)
        self.manifest(self.home / "projects/b-demo", kit)
        result = self.discover()
        self.assertTrue(result["ok"], result)
        self.assertEqual(Path(result["selected"]["kit"]["path"]), kit)

    def test_incomplete_embedded_kit_is_not_selected_as_standalone_kit(self):
        isaac = self.isaac(self.home / "isaacsim-6.0.1")
        (isaac / "kit/extscore/omni.kit.loop-default/config/extension.toml").unlink()
        result = self.discover()
        self.assertFalse(result["ok"])
        self.assertIn("isaac", result["selected"])
        self.assertNotIn("kit", result["selected"])


if __name__ == "__main__":
    unittest.main()
