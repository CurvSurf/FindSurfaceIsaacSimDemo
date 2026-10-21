"""Dark-only appearance and compatibility with previously saved theme settings."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/extensions/com.curvsurf.findsurface_isaac_sim"
sys.path.insert(0, str(PACKAGE))

from findsurface_isaac_sim.command_facade import session_from_config
from findsurface_isaac_sim.config import DEFAULT_APP_CONFIG, load_app_config, write_app_config
from findsurface_isaac_sim.render_layers import RenderTheme, heatmap_color, style_for_theme
from findsurface_isaac_sim.user_preferences import load_user_preferences


class DarkAppearanceTests(unittest.TestCase):
    def test_only_dark_render_style_is_supported(self):
        self.assertEqual(tuple(RenderTheme), (RenderTheme.EDITOR_DARK,))
        style = style_for_theme(RenderTheme.EDITOR_DARK)
        self.assertEqual(style.background, (0.035, 0.05, 0.075, 1.0))
        self.assertEqual(heatmap_color(style, 0.5), (1.0, 1.0, 0.0, 1.0))

    def test_legacy_light_config_is_ignored_and_not_written_back(self):
        config = copy.deepcopy(DEFAULT_APP_CONFIG)
        config["theme"] = "classic-light"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_app_config(path)
            self.assertNotIn("theme", loaded.config)
            self.assertFalse(hasattr(session_from_config(loaded.config), "theme"))
            write_app_config(path, config)
            self.assertNotIn("theme", json.loads(path.read_text(encoding="utf-8")))

    def test_legacy_preference_does_not_restore_light(self):
        loaded = load_user_preferences(DEFAULT_APP_CONFIG, json.dumps({
            "schema_version": 1, "theme": "classic-light", "display": {"point_size": 3.0},
        }))
        self.assertTrue(loaded.loaded, loaded.diagnostic)
        self.assertNotIn("theme", loaded.config)
        self.assertEqual(session_from_config(loaded.config).interaction.point_rendering_size, 3.0)

    def test_ui_and_dispatch_no_longer_offer_theme_changes(self):
        for name in ("classic_ui.py", "extension.py", "ui_shell.py"):
            source = (PACKAGE / "findsurface_isaac_sim" / name).read_text(encoding="utf-8")
            self.assertNotIn("settings.theme", source)
            self.assertNotIn("classic-light", source)
            self.assertNotIn("state.theme", source)


if __name__ == "__main__":
    unittest.main()
