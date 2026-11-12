"""Kit --exec entry point for the portable validation kit (no publishing)."""
import asyncio
import gc
import importlib
import json
from pathlib import Path
import time
import traceback

import carb.settings
import omni.kit.app


async def check():
    app = omni.kit.app.get_app()
    settings = carb.settings.get_settings()
    config = json.loads(Path(settings.get_as_string("/findsurface/validation/config")).read_text())
    result = {"phase": config["phase"], "case": config["case"], "ok": False}
    instance = None
    try:
        # Record the observed value only; its later value does not establish
        # whether a startup dialog appeared. Arguments/app configs carry the request.
        result["startup_dialog_setting_after_startup"] = settings.get_as_bool("/persistent/renderer/startupMessageDisplayed")
        manager = app.get_extension_manager()
        for _ in range(180 if config["phase"] == "child" else 10):
            await app.next_update_async()
        if config["phase"] == "child":
            assert manager.is_extension_enabled(config["main"]), "Demo implementation did not start"
            assert not manager.is_extension_enabled(config["launcher"]), "Launcher leaked into child"
            if config["case"] == "isaac":
                instance = next((obj for obj in gc.get_objects() if type(obj).__name__ == "FindSurfaceIsaacSimExtension"), None)
                assert instance is not None, "Isaac demo instance missing"
                deadline = time.monotonic() + 120
                while instance._scene_task is not None and not instance._scene_task.done() and time.monotonic() < deadline:
                    await app.next_update_async()
                assert instance._scene_session is not None, "Isaac scene failed to initialize"
                if instance._scene_task is not None and instance._scene_task.done():
                    instance._scene_task.result()
            from findsurface.bootstrap import FindSurfaceDetector
            detector = FindSurfaceDetector()
            detector.close()
            picker = importlib.import_module(config["module"] + ".native_picking")
            picker.NativePointPicker()
            result.update(native_sdk_loaded=True, native_picker_loaded=True,
                          scene_initialized=config["case"] == "isaac")
        elif config["phase"] == "verify":
            from omni.kit.registry.nucleus.verification import (
                verify_and_get_package_id_for_publishing, verify_extension_for_publishing,
            )
            settings.set_bool("/app/extensions/skipPublishVerification", False)
            assert manager.set_extension_enabled_immediate(config["extension"], True)
            metadata = manager.get_extension_dict(manager.get_enabled_extension_id(config["extension"]))
            result["package_id"] = verify_and_get_package_id_for_publishing(metadata)
            assert verify_extension_for_publishing(metadata)[0], "Kit metadata verification failed"
            result["published"] = False
        else:
            from findsurface.bootstrap import demo_launcher_ui
            from findsurface.bootstrap.demo_launcher import make_launch_plan
            assert manager.is_extension_enabled(config["launcher"])
            assert not manager.is_extension_enabled(config["main"]), "Demo enabled in host"
            windows = [obj for obj in gc.get_objects() if type(obj) is demo_launcher_ui.DemoLauncherWindow]
            assert len(windows) == 1, "Expected one launcher window"
            window = windows[0]
            assert window.root == Path(config["launcher_root"]).resolve(), "Wrong launcher package"
            assert not window.child.running, "Activation started a demo without a click"
            window.runtime.set_value(config["runtime"])
            extra = ["--exec", config["smoke_script"], "--/findsurface/validation/config=" + config["child_config"],
                     "--/log/file=" + config["child_log"], "--/app/settings/persistent=0",
                     "--/app/settings/loadUserConfig=0", "--/app/extensions/registryEnabled=0",
                     "--/persistent/renderer/startupMessageDisplayed=true"]
            demo_launcher_ui.make_launch_plan = lambda root, requested: make_launch_plan(root, requested, extra)
            window.launch()
            assert window.child.running, window.status.text
            result["child_pid"] = window.child.process.pid
            manager.set_extension_enabled_immediate(config["launcher"], False)
            assert window.child.running, "Disabling launcher stopped the demo"
            deadline = time.monotonic() + 180
            while window.child.running and time.monotonic() < deadline:
                await app.next_update_async()
                await asyncio.sleep(0.05)
            assert not window.child.running, "Child timed out"
            result["child_exit"] = window.child.process.returncode
            child = json.loads(Path(config["child_result"]).read_text())
            assert child.get("ok") and result["child_exit"] == 0, child
            assert not manager.is_extension_enabled(config["main"])
            result.update(activation_only_opened_launcher=True, disable_kept_child_alive=True,
                          demo_not_enabled_in_host=True, child=child)
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    Path(config["result"]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if instance is not None:
        instance.request_safe_exit()
    else:
        app.post_quit(0 if result["ok"] else 1)


asyncio.ensure_future(check())
