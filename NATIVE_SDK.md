# Native FindSurface SDK setup

FindSurface detection requires a separately distributed native SDK. Neither
`FindSurface.dll` nor `libFindSurface.so` is included in this repository.

## Setup and launch

Use `Start_Demo.bat` on Windows or `bash Start_Demo.sh` on Ubuntu for dependency setup.
The launcher obtains the newest explicitly approved compatible SDK from
[FindSurfaceOmniverseExtension releases](https://github.com/CurvSurf/FindSurfaceOmniverseExtension/releases).
The current approved version is **v1.0.1**: `FindSurface.dll` on Windows x86_64,
or `libFindSurface.so` on Ubuntu x86_64. A fresh download is stored in the
installation's private folder and passed to this app by its exact path.

An existing Windows SDK at
`source/extensions/com.findsurface.bootstrap/data/native/windows-x86_64/FindSurface.dll`
is reused and is not removed by `Remove_Demo.bat`.

For offline Ubuntu x86_64 setup, place the **approved Linux build** at
`source/extensions/com.findsurface.bootstrap/data/native/linux-x86_64/libFindSurface.so`,
or pass `bash Start_Demo.sh --sdk-path /path/to/libFindSurface.so`.
This explicit external library is reused, load-tested and preserved on removal.
SDK binaries must not be committed to the source repository. The bundled C++
point picker is a separate, product-owned library; Ubuntu setup builds it with CMake.
Placing the SDK here before the extension submodule has been initialized is
supported: the launcher backs up the pre-existing files and restores the SDK
after initialization, including after a failed Git attempt. There is no need
to empty this directory or delete your library before retrying.

## Compatible updates

`tools/sdk_releases.json` lists the approved releases and each platform's pinned
SHA-256. Compatibility is explicitly approved, rather than inferred from whatever
GitHub currently marks as latest. When a recognized older SDK or an app-owned
private SDK can be upgraded, the launcher asks `Update now? [y/N]`.
Declining keeps the installed file. Unknown/custom libraries and an explicit
Ubuntu `--sdk-path` are preserved without update prompts.

An accepted update downloads to a temporary file, verifies SHA-256 and checks
the native API and context creation in a separate process before atomically
replacing the old file. A failed update preserves the existing binary; a failed
fresh installation stops with an error. An SDK loaded by another app may prevent
replacement on Windows; close that app and retry. Update receipts live under
`.installation/sdk-updates`; no binaries are committed to Git.

If the approved v1.0.1 release is not published (HTTP 404), fresh setup can fall
back to the retained, checksum-pinned v1.0.0. Other network or verification
failures do not silently substitute another version. Existing latest-version
SDKs are recognized by their checksum without a network request.

## Offline setup

Place the approved SDK at the path above. Alternatively, with an installed
Isaac Sim runtime, supply its path explicitly:

```powershell
.\repo.bat launch -- --/exts/com.curvsurf.findsurface_isaac_sim/findSurfaceLibraryPath=C:/path/to/FindSurface.dll
```

The direct launcher disables the common extension's automatic SDK download.
Use `Start_Demo.bat` or `Start_Demo.sh` for verified acquisition, compatible
updates and ownership-aware setup. The developer `repo.bat` / `repo.sh` entrypoints
do not prepare dependencies or offer SDK updates.
An unavailable or unloadable SDK leaves detection unavailable; it does not remove
Isaac Sim's GPU and driver requirements. See [INSTALL.md](INSTALL.md) for setup,
dependency reuse, and safe removal.

Use and redistribution of the SDK follow CurvSurf's SDK terms and any applicable
commercial agreement.
