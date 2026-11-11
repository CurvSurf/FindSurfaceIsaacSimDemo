from pathlib import Path

import omni.ext
from findsurface.bootstrap.demo_launcher_ui import DemoLauncherWindow


class DemoLauncherExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self.launcher = DemoLauncherWindow(Path(__file__).resolve().parents[1])

    def on_shutdown(self):
        launcher, self.launcher = getattr(self, "launcher", None), None
        if launcher is not None:
            launcher.close()
