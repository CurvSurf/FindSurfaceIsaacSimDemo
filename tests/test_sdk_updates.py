"""Exercise SDK upgrades without contacting GitHub or installing native code."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import urllib.error
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sdk_updater", ROOT / "tools/update_sdk.py")
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class SdkUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="fsod-sdk-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sdk = self.root / "FindSurface.dll"
        self.state = self.root / "state"
        self.old, self.new = b"old-sdk-fixture", b"new-sdk-fixture"
        self.sdk.write_bytes(self.old)
        self.policy = json.loads(updater.POLICY_PATH.read_text(encoding="utf-8"))
        for entry, data in zip(self.policy["releases"], (self.old, self.new)):
            for asset in entry["assets"].values():
                asset["sha256"] = digest(data)
        self.key = "windows-x86_64"
        self.metadata = {
            "tag_name": "v1.0.1", "draft": False, "prerelease": False,
            "html_url": "https://github.com/CurvSurf/FindSurfaceOmniverseExtension/releases/tag/v1.0.1",
            "body": "Compatible bug fixes.",
            "assets": [{
                "name": "FindSurface.dll", "digest": "sha256:" + digest(self.new),
                "browser_download_url": "https://github.com/CurvSurf/FindSurfaceOmniverseExtension/releases/download/v1.0.1/FindSurface.dll",
            }],
        }
        self.api = self.patch("get_release", return_value=self.metadata)
        self.download = self.patch("download", side_effect=lambda url, path: path.write_bytes(self.new))
        self.validate = self.patch("validate_library")
        self.patch("print")
        self.confirm = mock.Mock(return_value=True)

    def patch(self, name, **kwargs):
        patcher = mock.patch.object(updater, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def run_update(self, **kwargs):
        return updater.ensure_sdk(self.sdk, self.key, policy=self.policy,
                                  confirm=self.confirm, state_dir=self.state, **kwargs)

    def test_success_replaces_original_without_backup_or_temporary_binary(self):
        self.run_update()
        self.assertEqual(self.sdk.read_bytes(), self.new)
        self.assertEqual(list(self.root.glob("*.download")), [])
        self.assertEqual(list(self.root.glob("*.bak")), [])
        self.assertEqual(len(list(self.state.glob("*.json"))), 1)
        self.assertEqual(self.confirm.call_args.args[:2], ("v1.0.0", "v1.0.1"))
        self.validate.assert_called_once()
        self.api.assert_called_once_with(self.policy["repository"], "v1.0.1")

    def test_decline_keeps_original_without_download_or_load(self):
        self.confirm.return_value = False
        self.run_update()
        self.assertEqual(self.sdk.read_bytes(), self.old)
        self.download.assert_not_called()
        self.validate.assert_not_called()

    def test_latest_binary_does_not_query_network_or_prompt(self):
        self.sdk.write_bytes(self.new)
        self.run_update()
        self.api.assert_not_called()
        self.confirm.assert_not_called()

    def test_unknown_user_binary_is_not_adopted_or_replaced(self):
        self.sdk.write_bytes(b"custom SDK")
        self.run_update()
        self.assertEqual(self.sdk.read_bytes(), b"custom SDK")
        self.api.assert_not_called()
        self.assertFalse(self.state.exists())

    def test_managed_binary_can_be_updated_after_installer_ownership_validation(self):
        self.sdk.write_bytes(b"previous app-private SDK")
        self.run_update(managed=True)
        self.assertEqual(self.sdk.read_bytes(), self.new)

    def test_prepublication_404_leaves_installed_sdk_usable(self):
        self.api.side_effect = urllib.error.HTTPError("https://example.invalid", 404, "Not Found", {}, None)
        self.run_update()
        self.assertEqual(self.sdk.read_bytes(), self.old)
        self.confirm.assert_not_called()

    def test_prepublication_fresh_install_uses_retained_pinned_v100(self):
        self.sdk.unlink()
        self.api.side_effect = urllib.error.HTTPError("https://example.invalid", 404, "Not Found", {}, None)
        self.download.side_effect = lambda url, path: path.write_bytes(self.old)
        self.run_update()
        self.assertEqual(self.sdk.read_bytes(), self.old)
        self.assertIn("/v1.0.0/", self.download.call_args.args[0])
        self.confirm.assert_not_called()

    def test_fresh_install_after_publication_uses_v101(self):
        self.sdk.unlink()
        self.run_update()
        self.assertEqual(self.sdk.read_bytes(), self.new)
        self.confirm.assert_not_called()

    def test_unavailable_network_and_download_error_preserve_existing(self):
        for stage in (self.api, self.download):
            with self.subTest(stage=stage):
                stage.side_effect = OSError("network unavailable")
                self.run_update()
                self.assertEqual(self.sdk.read_bytes(), self.old)
                self.assertEqual(list(self.root.glob("*.download")), [])
                stage.side_effect = None

    def test_no_binary_and_download_failure_is_reported_as_failure(self):
        self.sdk.unlink()
        self.download.side_effect = OSError("download failed")
        with self.assertRaises(OSError):
            self.run_update()
        self.assertFalse(self.sdk.exists())

    def test_digest_failure_preserves_original_and_removes_candidate(self):
        self.download.side_effect = lambda url, path: path.write_bytes(b"corrupted")
        self.run_update()
        self.assertEqual(self.sdk.read_bytes(), self.old)
        self.assertEqual(list(self.root.glob("*.download")), [])
        self.validate.assert_not_called()

    def test_abi_failure_preserves_original(self):
        self.validate.side_effect = ValueError("missing fs_find_torus")
        self.run_update()
        self.assertEqual(self.sdk.read_bytes(), self.old)
        self.assertEqual(list(self.root.glob("*.download")), [])

    def test_locked_destination_preserves_original_and_cleans_candidate(self):
        with mock.patch.object(updater.os, "replace", side_effect=PermissionError("DLL is in use")):
            self.run_update()
        self.assertEqual(self.sdk.read_bytes(), self.old)
        self.assertEqual(list(self.root.glob("*.download")), [])

    def test_external_change_during_download_is_not_overwritten(self):
        def changed(url, path):
            path.write_bytes(self.new)
            self.sdk.write_bytes(b"changed by another writer")
        self.download.side_effect = changed
        self.run_update()
        self.assertEqual(self.sdk.read_bytes(), b"changed by another writer")

    def test_busy_updater_does_not_prompt_or_replace(self):
        self.state.mkdir()
        identity = hashlib.sha256(updater.os.path.normcase(str(self.sdk)).encode("utf-8")).hexdigest()
        with updater.sdk_lock(self.state / (identity + ".lock")):
            self.run_update()
        self.confirm.assert_not_called()
        self.assertEqual(self.sdk.read_bytes(), self.old)

    def test_unexpected_url_missing_digest_or_changed_pinned_binary_is_rejected(self):
        cases = [copy.deepcopy(self.metadata) for _ in range(3)]
        cases[0]["assets"][0]["browser_download_url"] = "https://example.invalid/sdk.dll"
        cases[1]["assets"] = []
        cases[2]["assets"][0]["digest"] = "sha256:" + "0" * 64
        for metadata in cases:
            with self.subTest(metadata=metadata):
                self.api.return_value = metadata
                self.run_update()
                self.assertEqual(self.sdk.read_bytes(), self.old)
        self.download.assert_not_called()

    def test_linux_update_uses_published_digest_and_receipt(self):
        self.key = "linux-x86_64"
        self.policy["releases"][-1]["assets"][self.key]["sha256"] = None
        asset = self.metadata["assets"][0]
        asset["name"] = "libFindSurface.so"
        asset["browser_download_url"] = asset["browser_download_url"].replace("FindSurface.dll", "libFindSurface.so")
        self.run_update()
        self.assertEqual(self.sdk.read_bytes(), self.new)
        self.api.reset_mock()
        self.run_update()
        self.api.assert_not_called()

    def test_linux_release_without_digest_does_not_replace_original(self):
        self.key = "linux-x86_64"
        self.policy["releases"][-1]["assets"][self.key]["sha256"] = None
        asset = self.metadata["assets"][0]
        asset.update(name="libFindSurface.so", digest=None,
                     browser_download_url=asset["browser_download_url"].replace("FindSurface.dll", "libFindSurface.so"))
        self.run_update()
        self.assertEqual(self.sdk.read_bytes(), self.old)
        self.download.assert_not_called()

    def test_noninteractive_prompt_defaults_to_no(self):
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertFalse(updater.ask_update("v1.0.0", "v1.0.1", self.sdk, self.metadata))
        with mock.patch("builtins.input", return_value=""):
            self.assertFalse(updater.ask_update("v1.0.0", "v1.0.1", self.sdk, self.metadata))

    def test_real_prompt_routes_yes_and_no_to_the_expected_actions(self):
        for answer in ("n", "y"):
            with self.subTest(answer=answer):
                self.sdk.write_bytes(self.old)
                with mock.patch("builtins.input", return_value=answer) as prompt:
                    updater.ensure_sdk(self.sdk, self.key, policy=self.policy, state_dir=self.state)
                prompt.assert_called_once_with("Update now? [y/N]: ")
                self.assertEqual(self.sdk.read_bytes(), self.new if answer == "y" else self.old)

    def test_remote_metadata_parser_rejects_unapproved_or_unstable_releases(self):
        # Exercise the production metadata parser rather than the mocked network lookup.
        parser_spec = importlib.util.spec_from_file_location("sdk_parser_test", ROOT / "tools/update_sdk.py")
        parser_module = importlib.util.module_from_spec(parser_spec)
        parser_spec.loader.exec_module(parser_module)
        for changes in ({"tag_name": "v2.0.0"}, {"draft": True}, {"prerelease": True}):
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({**self.metadata, **changes}).encode()
            with mock.patch.object(parser_module.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    parser_module.get_release(self.policy["repository"], "v1.0.1")


class SdkUpdateRoutingTests(unittest.TestCase):
    def test_source_and_packaged_entrypoints_include_updater(self):
        for script in ("tools/manage_installation.ps1", "tools/manage_installation.py"):
            self.assertIn("update_sdk", (ROOT / script).read_text(encoding="utf-8"), script)



class NativeSdkReplacementTests(unittest.TestCase):
    def test_real_candidate_validation_and_atomic_replacement(self):
        candidate = os.environ.get("FINDSURFACE_TEST_NEW_SDK_PATH")
        old = Path(os.environ.get("FINDSURFACE_TEST_OLD_SDK_PATH", ""))
        if os.name != "nt" or not candidate or not old.is_file():
            self.skipTest("Requires an existing Windows SDK and FINDSURFACE_TEST_NEW_SDK_PATH")
        policy = json.loads(updater.POLICY_PATH.read_text(encoding="utf-8"))
        new_hash = policy["releases"][-1]["assets"]["windows-x86_64"]["sha256"]
        self.assertEqual(updater.sha256(candidate), new_hash)
        updater.validate_library(old)
        with tempfile.TemporaryDirectory(prefix="fsod-native-sdk-test-") as directory:
            root = Path(directory)
            destination = root / "FindSurface.dll"
            shutil.copyfile(old, destination)
            metadata = {
                "html_url": "https://github.com/CurvSurf/FindSurfaceOmniverseExtension/releases/tag/v1.0.1",
                "assets": [{"name": "FindSurface.dll", "digest": "sha256:" + new_hash,
                    "browser_download_url": "https://github.com/CurvSurf/FindSurfaceOmniverseExtension/releases/download/v1.0.1/FindSurface.dll"}],
            }
            def copy_download(_url, path):
                self.assertEqual(updater.sha256(destination), policy["releases"][0]["assets"]["windows-x86_64"]["sha256"])
                shutil.copyfile(candidate, path)
            with mock.patch.object(updater, "get_release", return_value=metadata), \
                    mock.patch.object(updater, "download", side_effect=copy_download), \
                    mock.patch("builtins.input", return_value="y") as prompt, redirect_stdout(io.StringIO()) as output:
                updater.ensure_sdk(destination, "windows-x86_64", policy=policy, state_dir=root / "state")
            self.assertIn("v1.0.0 -> v1.0.1", output.getvalue())
            prompt.assert_called_once()
            self.assertEqual(updater.sha256(destination), new_hash)
            updater.validate_library(destination)
            self.assertEqual(list(root.glob("*.download")), [])
            self.assertEqual(list(root.glob("*.bak")), [])


if __name__ == "__main__":
    unittest.main()
