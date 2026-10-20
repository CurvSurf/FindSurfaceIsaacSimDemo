"""Exercise SDK-before-submodule setup with real local Git clones, without a network."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import repo
import manage_installation as installer

RELATIVE = 'source/extensions/com.findsurface.bootstrap'


@unittest.skipUnless(shutil.which('git'), 'Git is required for local clone regression tests')
class BootstrapInitializationTests(unittest.TestCase):
    def git(self, directory, *arguments, check=True):
        return subprocess.run(['git', '-c', f'safe.directory={directory.as_posix()}',
                               '-c', 'user.name=Installation Test', '-c', 'user.email=test@example.invalid',
                               '-C', str(directory), *arguments], check=check,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.environment = patch.dict(os.environ, {'GIT_ALLOW_PROTOCOL': 'file'})
        self.environment.start()
        self.library = self.base / 'library origin'
        self.library.mkdir()
        self.git(self.library, 'init')
        (self.library / 'config').mkdir()
        (self.library / 'config/extension.toml').write_text('[package]\n', encoding='utf-8')
        self.git(self.library, 'add', '.')
        self.git(self.library, 'commit', '-m', 'Fixture extension')
        origin = self.base / 'app origin'
        origin.mkdir()
        self.git(origin, 'init')
        self.git(origin, 'submodule', 'add', str(self.library), RELATIVE)
        self.git(origin, 'commit', '-am', 'Pin fixture extension')
        self.project = self.base / 'fresh checkout with spaces'
        self.git(self.base, 'clone', '--no-local', str(origin), str(self.project))
        self.bootstrap = self.project / RELATIVE
        self.sdk = self.bootstrap / 'data/native/linux-x86_64/libFindSurface.so'
        self.sdk.parent.mkdir(parents=True, exist_ok=True)
        self.sdk.write_bytes(b'pre-positioned approved SDK fixture')
        self.dll = self.bootstrap / 'data/native/windows-x86_64/FindSurface.dll'
        self.dll.parent.mkdir(parents=True)
        self.dll.write_bytes(b'Windows SDK fixture')

    def tearDown(self):
        self.environment.stop()
        # Git object files can be read-only on Windows; reset fixture attributes only.
        def writable_retry(function, path, _error):
            os.chmod(path, 0o700)
            function(path)
        shutil.rmtree(self.base, onerror=writable_retry)
        self.temp.cleanup()

    def assert_sdk_preserved(self):
        self.assertEqual(self.sdk.read_bytes(), b'pre-positioned approved SDK fixture')
        self.assertEqual(self.dll.read_bytes(), b'Windows SDK fixture')

    def test_reproduced_nonempty_clone_failure_then_safe_initialization(self):
        failed = self.git(self.project, 'submodule', 'update', '--init', '--recursive', check=False)
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse((self.bootstrap / 'config/extension.toml').exists())
        self.assert_sdk_preserved()
        self.assertEqual(repo.initialize_dependencies(self.project), 0)
        self.assertTrue((self.bootstrap / 'config/extension.toml').is_file())
        self.assert_sdk_preserved()
        # Already initialized reruns are no-ops, not another backup or overwrite.
        self.assertEqual(repo.initialize_dependencies(self.project), 0)
        self.assert_sdk_preserved()

    def test_failed_fetch_restores_sdk_to_original_path_and_retry_succeeds(self):
        self.git(self.project, 'config', f'submodule.{RELATIVE}.url', str(self.base / 'missing origin'))
        self.assertNotEqual(repo.initialize_dependencies(self.project), 0)
        self.assert_sdk_preserved()
        self.assertTrue(list((self.project / '.artifacts').glob('bootstrap-backup-*/bootstrap')))
        self.git(self.project, 'config', f'submodule.{RELATIVE}.url', str(self.library))
        self.assertEqual(repo.initialize_dependencies(self.project), 0)
        self.assert_sdk_preserved()

    def test_ubuntu_start_uses_same_preserving_initializer(self):
        (self.project / 'tools').mkdir()
        installer.write_json(self.project / 'tools/installation_dependencies.json', {
            'linux-x86_64': {}})
        args = argparse.Namespace(sdk_path=None, install_base=str(self.base / 'private'), isaac_root=None)
        with patch.object(installer.shutil, 'which', return_value='fixture-tool'), \
                patch.object(repo, 'initialize_dependencies', wraps=repo.initialize_dependencies) as initialize, \
                patch.object(installer, 'find_runtime', side_effect=RuntimeError('stop before runtime download')):
            with self.assertRaisesRegex(RuntimeError, 'stop before runtime download'):
                installer.start(args, installer.Installation(self.project))
            initialize.assert_called_once_with(self.project)
        self.assertTrue((self.bootstrap / 'config/extension.toml').is_file())
        self.assert_sdk_preserved()

    def test_non_submodule_directory_is_never_moved(self):
        self.git(self.project, 'rm', '--cached', '-f', RELATIVE)
        with self.assertRaisesRegex(repo.RuntimeDiscoveryError, 'not a pinned submodule'):
            repo.initialize_dependencies(self.project)
        self.assert_sdk_preserved()
        self.assertFalse((self.project / '.artifacts').exists())


if __name__ == '__main__':
    unittest.main()
