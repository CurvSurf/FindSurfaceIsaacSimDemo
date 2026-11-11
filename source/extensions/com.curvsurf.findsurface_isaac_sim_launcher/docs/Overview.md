# FindSurface Isaac Sim Demo Launcher

Enable this extension to open the launcher. Click **Run Demo** to start the
existing demo in a separate process/window. The demo is not enabled inside
the host. Disabling the launcher does not stop an already running demo.

## Requirements and usage

- Launcher host: Kit 110.1.2 / Python 3.12, Windows x86-64 (tested).
- Installed demo runtime: **isaac 6.0.1**. No runtime is included or installed.
- Registry dependency: `com.findsurface.bootstrap` **1.1.0** exactly.

The launcher checks the runtime version. If automatic discovery fails, enter
the installed runtime folder and click **Run Demo**. The archive bundles the
demo experience, code, resources and native libraries; no source checkout is
required. One child can be running per launcher instance. GPU/driver and OS
requirements are those of the selected runtime. Runtime extension/pip download
requirements remain the runtime's responsibility; this is not an offline runtime.

If the child exits during startup, consult its Kit logs. Existing
`Start_Demo.bat` / `Start_Demo.sh` workflows still start the demo directly.

Manual and support: https://github.com/CurvSurf/FindSurfaceIsaacSimDemo

## Terms

Retain packaged licenses and third-party notices. Native FindSurface SDK v1.0.1
is subject to separate CurvSurf SDK terms and applicable commercial agreements;
inclusion grants no additional SDK rights. See https://github.com/CurvSurf/FindSurface.
Community packages are not supported or endorsed by NVIDIA.
