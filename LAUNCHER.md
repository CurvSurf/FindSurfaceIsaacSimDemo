# Demo launcher structure

The core `com.findsurface.bootstrap` remains a reusable SDK extension.
`com.curvsurf.findsurface_isaac_sim_launcher` opens a small launcher window. It does **not** depend on or enable
`com.curvsurf.findsurface_isaac_sim` in the host. **Run Demo** starts the existing experience in a
separate process, after checking for isaac 6.0.1. Disabling the launcher
or closing its host leaves an already running demo open. The same launcher
rejects another launch while its child is running.

## Local use

`Start_Demo.bat` / `Start_Demo.sh` still start the demo directly, including the
existing installation preparation. No launcher window is added to that path.

To preview the new launcher after preparing the existing local dependencies:

```text
python tools/launch_launcher.py
python tools/launch_launcher.py --runtime-root "path/to/installed/runtime"
```

Or add `source/extensions` to a compatible Kit app's extension search folders and
enable `com.curvsurf.findsurface_isaac_sim_launcher` in Extension Manager. Click **Run Demo** in the launcher.
If automatic discovery cannot find the runtime, enter its folder in the window.
This step does not download or install a runtime.

The standalone demo remains the source app at
`source/apps/com.curvsurf.findsurface_isaac_sim_demo.kit`. Its detection, UI, and shutdown code is unchanged.

## Local payload staging

```text
python tools/prepare_launcher.py --destination .artifacts/launcher-local/com.curvsurf.findsurface_isaac_sim_launcher
```

Use a fresh destination for each staging run. This creates an extension folder
containing `data/payload/apps` and `data/payload/extensions`, including the core,
demo code, presets, and locally prepared native binaries. It copies the current
source app definition; there is no second maintained app definition.

To inspect the staged launcher, add the destination's **parent** folder to the
host's extension folders and make the shared core extension available to the
host. The host core supplies the launcher UI/backend; the payload core is used
by the child. The child receives the staged extension folder explicitly, so it
does not require the original demo checkout. Configure an installed runtime in
the launcher window.

Staging creates a local launcher folder for testing.
