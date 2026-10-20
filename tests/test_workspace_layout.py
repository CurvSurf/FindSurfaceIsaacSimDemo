import ast
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'source/extensions/com.curvsurf.findsurface_isaac_sim/findsurface_isaac_sim'
sys.path.insert(0, str(PACKAGE.parent))
from findsurface_isaac_sim.workspace_layout import (
    PROPERTY_TITLES, decode_workspace_layout, encode_workspace_layout,
    fit_floating_bounds, fit_workspace_layout,
)


def layout(floating=None):
    return {'side_ratio': .29, 'tab_order': list(PROPERTY_TITLES),
            'active_tab': 'Scanner', 'floating': floating or {}}


def load_kit_method(file, class_name, method, namespace):
    """Execute the actual method body with SDK boundaries replaced, not a copy."""
    tree = ast.parse((PACKAGE / file).read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = copy.deepcopy(next(node for node in cls.body if getattr(node, 'name', None) == method))
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(PACKAGE / file), 'exec'), namespace)
    return namespace[method]


class WorkspaceLayoutTests(unittest.TestCase):
    def test_shift_preserves_size_on_all_edges(self):
        for source, expected in (([-477.33, -6.67, 490, 680], [0, 0, 490, 680]),
                                 ([1000, 700, 490, 680], [790, 120, 490, 680]),
                                 ([10, 20, 490, 680], [10, 20, 490, 680])):
            self.assertEqual(fit_floating_bounds(source, (1280, 800)), expected)

    def test_oversized_window_is_fully_contained(self):
        self.assertEqual(fit_floating_bounds([-300, 500, 2000, 1400], (1280, 800)), [0, 0, 1280, 800])
        self.assertEqual(fit_floating_bounds([20, 20, 600, 300], (500, 800)), [0, 20, 500, 300])

    def test_tiny_window_grows_to_reasonable_minimum(self):
        self.assertEqual(fit_floating_bounds([1200, 780, 2, 3], (1280, 800)), [960, 560, 320, 240])
        # Containment takes precedence if the OS content itself is unusually small.
        saved = fit_floating_bounds([100, 100, 10, 10], (200, 100))
        self.assertEqual(saved, [0, 0, 200, 100])
        self.assertEqual(fit_floating_bounds(saved, (1280, 800)), [0, 0, 320, 240])

    def test_save_is_idempotent_and_does_not_move_live_layout(self):
        current = layout({'Settings': [-477.33, -6.67, 490, 680]})
        original = copy.deepcopy(current)
        saved = fit_workspace_layout(current, (1280, 800))
        self.assertEqual(current, original)
        self.assertEqual(saved['floating']['Settings'], [0, 0, 490, 680])
        self.assertEqual(fit_workspace_layout(saved, (1280, 800)), saved)
        self.assertEqual(decode_workspace_layout(encode_workspace_layout(saved)), saved)

    def test_legacy_tiny_and_offscreen_values_can_be_repaired_but_nonfinite_cannot(self):
        old = layout({'Settings': [-477.33, -6.67, 490, 680], 'Camera': [99999, 99999, 3, 4]})
        decoded = decode_workspace_layout(json.dumps({'version': 2, 'layout': old}))
        self.assertEqual(fit_workspace_layout(decoded, (800, 600))['floating']['Camera'], [480, 360, 320, 240])
        for value in (float('nan'), float('inf'), -float('inf')):
            invalid = layout({'Settings': [value, 0, 490, 680]})
            self.assertIsNone(decode_workspace_layout(json.dumps({'version': 2, 'layout': invalid})))
        for extent in ((0, 800), (1280, 0), (float('nan'), 800)):
            with self.assertRaises(ValueError):
                fit_workspace_layout(layout(), extent)

    def test_capture_uses_full_logical_os_content_not_viewport_or_physical_pixels(self):
        from enum import Enum
        class Tab(Enum):
            SETTINGS = 'Settings'
        window = SimpleNamespace(docked=False, dock_order=0, position_x=-477.33,
                                 position_y=-6.67, width=490, height=680)
        host = SimpleNamespace(property_windows={Tab.SETTINGS: window}, _layout_preference=None,
                               workspace_content_size=lambda: (1280, 800))
        capture = load_kit_method('classic_ui.py', 'ClassicUiChrome', 'capture_layout', {
            'SIDE_PANEL_DOCK_RATIO': .29, 'fit_workspace_layout': fit_workspace_layout})
        self.assertEqual(capture(host, SimpleNamespace(width=600))['floating']['Settings'], [0, 0, 490, 680])
        self.assertEqual(window.position_x, -477.33)
        # A user can leave it outside during this run; only the saved copy moves.


class DockRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_incomplete_initial_docking_forces_default_and_clears_saved_preference(self):
        settings = Mock()
        namespace = {'asyncio': asyncio, 'carb': SimpleNamespace(
            log_warn=Mock(), log_error=Mock(), settings=SimpleNamespace(get_settings=lambda: settings)),
            'PERSISTENT_WORKSPACE_PATH': '/layout'}
        restore = load_kit_method('extension.py', 'FindSurfaceIsaacSimExtension', '_restore_workspace_after_dock', namespace)
        async def reset(_window):
            host._chrome._layout_preference = None
        host = SimpleNamespace(_chrome=SimpleNamespace(dock_state=lambda: {'Settings': False},
                               reset_docking=reset, _layout_preference={'bad': True}), _window=object(),
                               _saved_workspace_layout={'bad': True}, _workspace_layout_cache={'bad': True},
                               _user_preferences_enabled=True)
        def capture(): host._workspace_layout_cache = layout()
        host._capture_workspace_layout = capture
        await restore(host, None)
        self.assertIsNone(host._saved_workspace_layout)
        self.assertIsNone(host._chrome._layout_preference)
        self.assertEqual(host._default_workspace_layout, layout())
        settings.set.assert_called_once_with('/layout', '')

    async def test_restore_error_also_forces_default_without_writing_disabled_preferences(self):
        settings = Mock()
        namespace = {'asyncio': asyncio, 'carb': SimpleNamespace(log_warn=Mock(), log_error=Mock(),
            settings=SimpleNamespace(get_settings=lambda: settings)), 'PERSISTENT_WORKSPACE_PATH': '/layout'}
        restore = load_kit_method('extension.py', 'FindSurfaceIsaacSimExtension', '_restore_workspace_after_dock', namespace)
        async def broken(): raise RuntimeError('injected floating restore error')
        reset = Mock()
        async def reset_default(_window): reset()
        host = SimpleNamespace(_chrome=SimpleNamespace(dock_state=lambda: {'Settings': True},
            capture_layout=lambda _: layout(), restore_floating_layout=broken, reset_docking=reset_default),
            _window=object(), _saved_workspace_layout={'bad': True}, _workspace_layout_cache=None,
            _user_preferences_enabled=False)
        host._capture_workspace_layout = lambda: setattr(host, '_workspace_layout_cache', layout())
        await restore(host, None)
        reset.assert_called_once()
        settings.set.assert_not_called()

    def test_minimized_shutdown_keeps_last_bounded_snapshot(self):
        settings = Mock()
        persist = load_kit_method('extension.py', 'FindSurfaceIsaacSimExtension', '_persist_workspace_layout', {
            'carb': SimpleNamespace(log_warn=Mock(), settings=SimpleNamespace(get_settings=lambda: settings)),
            'encode_workspace_layout': encode_workspace_layout, 'PERSISTENT_WORKSPACE_PATH': '/layout'})
        cached = layout({'Settings': [0, 0, 490, 680]})
        host = SimpleNamespace(_user_preferences_enabled=True, _window=SimpleNamespace(visible=True),
                               _workspace_layout_cache=cached,
                               _capture_workspace_layout=Mock(side_effect=ValueError('zero minimized extent')))
        persist(host)
        self.assertEqual(decode_workspace_layout(settings.set.call_args.args[1]), cached)


if __name__ == '__main__':
    unittest.main()
