# Setup and removal

Use `Start_Demo.bat` on Windows or `bash Start_Demo.sh` on Ubuntu x86_64
from a recursive source clone or an assembled thin release
ZIP. It checks dependencies, prepares missing ones and runs the demo. It does
not clone/pull the app or install Git. A source checkout needs Git to initialize
missing submodules; GitHub's automatic Source ZIP omits them and cannot run as-is.
`Launch_Demo.bat` is a compatibility alias for `Start_Demo.bat`: both now prepare
dependencies before launch. Use `repo.bat launch` / `bash repo.sh launch` for a
developer launch without installation. The alias accepts the same PowerShell-style
options as `Start_Demo.bat`; pass direct-launch Kit arguments through `repo.bat`.

## Ubuntu setup

The launcher targets Ubuntu 22.04/24.04 **x86_64**, with Python 3.10 or newer.
Isaac Sim's [supported GPU, driver and operating-system requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/requirements.html)
still apply. Run from a terminal in a regular user's graphical desktop session;
do not use `sudo` to start the demo. WSL checks of the installer do not establish
that the application's GPU rendering and sensors work on a native Ubuntu desktop.

The installer uses Python's standard library. It needs `curl`; building an absent
native point picker also needs CMake and a C++ compiler. If these system tools are
missing, it stops with instructions. You can install them once with:

```bash
sudo apt update
sudo apt install python3 curl cmake build-essential
# Only for a source checkout with uninitialized submodules:
sudo apt install git
```

The scripts do not install system drivers/packages or change security policies.
They can also be run with `bash` after ZIP extraction, without setting execute bits:

```bash
bash Start_Demo.sh
bash Start_Demo.sh --prepare-only
bash Start_Demo.sh --isaac-root /opt/isaacsim-6.0.1
bash Start_Demo.sh --install-base "$HOME/DemoDependencies"
bash Start_Demo.sh --sdk-path /path/to/approved/libFindSurface.so
# Forward optional Kit settings after '--':
bash Start_Demo.sh -- --/app/window/width=1688
```

`FSIS_PYTHON` can select the host Python executable. The launcher reuses a complete
Isaac Sim 6.0.1 from the ownership record, `ISAACSIM_ROOT`, `~/isaacsim`,
`~/isaacsim-6.0.1`, `/opt/isaacsim`, or private installations under the recommended
base and chosen base. An invalid explicit `--isaac-root` fails instead of silently
installing a different runtime. Version, apps, executable and bundled Python are
checked before reuse. Library loading is checked through NVIDIA's `python.sh`,
which prepares its bundled Python environment.

When no reusable runtime exists, the script asks for the dependency location
(default `${XDG_DATA_HOME:-$HOME/.local/share}/CurvSurf/FindSurfaceIsaacSimDemo`)
and creates a unique `fsis-<installation-id>` folder. It asks for NVIDIA license
acceptance before downloading the official 6.0.1 Linux x86_64 archive. Budget
at least **50 GiB** free for the download, extraction and caches. Downloads resume
after interruption. Cached archives are rechecked before use. Linux archive
integrity uses NVIDIA's published **MD5** over HTTPS; this detects corruption,
but is not a cryptographic authenticity guarantee. The SDK requires a separately
pinned **SHA-256**. See the [official download page](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/download.html).

Extraction displays expanded bytes, percentage, elapsed time and estimated time
remaining. It preserves executable permissions and internal relative symlinks,
rejects traversal, escaping links and special files, and uses a new staging folder
on every attempt. A failed extraction is retained for inspection. Complete runtime
directories are never overwritten. The native C++ picker is built in the private
folder and reused until its source changes. Its path is passed only to the child
app through `FSIS_PICKING_LIBRARY_PATH`; nothing is registered globally.

The launcher downloads the approved **v1.0.1** Linux SDK from the
FindSurfaceOmniverseExtension GitHub release when no reusable library exists.
Recognized older SDKs can be updated after confirmation, with SHA-256 and native
API validation before replacement; see [NATIVE_SDK.md](NATIVE_SDK.md).
An explicit `--sdk-path` remains untouched. A Windows DLL cannot be used on Ubuntu.

### Ubuntu removal and verification

```bash
bash Remove_Demo.sh
```

Removal previews the exact app-private folder and preserved external dependencies,
requires typing `REMOVE`, checks the owner marker and checkout identity, and
refuses while Kit is running. It permanently deletes only that private folder.
Child symlinks are unlinked without deleting their targets. App source, exports,
user settings, reused external runtimes/SDKs and shared NVIDIA caches are preserved.
The `.installation` ownership record must stay with the original app folder;
the update/move rules below apply on both platforms.

Installer safety/reuse tests and the native picker build have been checked under
Ubuntu 24.04 in WSL. Native Ubuntu GPU startup, both sensor types, picking/capture,
and fresh runtime installation still require a compatible-machine acceptance run
before treating Ubuntu as release-validated. `--prepare-only` checks dependencies,
not the GPU/sensor workflow.

## Windows setup

## Source checkout and direct launch

Clone with submodules:

```powershell
git clone --recurse-submodules https://example.invalid/CurvSurf/FindSurfaceIsaacSimDemo.git
```

For an existing checkout, run `repo.bat init-dependencies` (Windows) or
`bash repo.sh init-dependencies` (Ubuntu) after pulling an
update that changes the extension pin. This initializes the recorded submodule
commit. Assembled release packages already include its source contents.

You may place the approved SDK in its checkout-local path before initializing
the extension. `Start_Demo` and `repo ... init-dependencies` preserve such files
under `.artifacts/bootstrap-backup-*`, initialize the pinned submodule, then
restore the native files to their original paths. They also restore the SDK
if Git initialization fails, so a retry does not require copying it again.
Other pre-existing files remain in the backup for inspection; no SDK is added
to Git. Use these entrypoints rather than a raw `git submodule update` when
the uninitialized submodule directory already contains local files.

For an already configured runtime, use `repo.bat launch` or `bash repo.sh launch`.
You can set `ISAACSIM_ROOT` or pass a runtime root explicitly:

```powershell
.\repo.bat doctor --isaac-sim-root C:\isaacsim-6.0.1
.\repo.bat launch --isaac-sim-root C:\isaacsim-6.0.1
```

This direct path uses Python 3 (`FSIS_PYTHON` can select its executable) and an
installed runtime. Use `Start_Demo.bat` for dependency preparation and launch.
It does not read the private SDK/picker paths from the installation record:
when using a private installation, prefer `Start_Demo.bat` / `bash Start_Demo.sh`,
or explicitly provide the SDK setting and `FSIS_PICKING_LIBRARY_PATH` yourself.

## First run

The following options and paths describe the Windows launcher; Ubuntu equivalents
are listed above.

The launcher asks for a private dependency base folder. Press Enter for
`%LOCALAPPDATA%\CurvSurf\FindSurfaceIsaacSimDemo`, or enter another writable folder.
A unique `fsis-<installation-id>` subfolder is created for this checkout. The
ownership record is `.installation/manifest.json` in the app folder (not in Git).
No administrator rights or global PATH/environment registration is required by
this launcher. Windows security policy may nevertheless restrict scripts/network.

Existing Isaac Sim **6.0.1** is discovered from `ISAACSIM_ROOT`, supported default
paths or `-IsaacRoot`. Its executable/Python/apps/version are checked, and the
existing installation is recorded as **external**, never owned by this app.
Discovery also checks immediate `fsis-<installation-id>\isaacsim-6.0.1` folders
under the recommended `%LOCALAPPDATA%\CurvSurf\FindSurfaceIsaacSimDemo` base and
the current private dependency base. Complete compatible runtimes are reused
even if the app folder's installation record was lost during replacement.
Other installations remain external: the new remover does not delete them.
Keep `.installation` when replacing files in the same app folder to preserve
the original ownership record and removal capability. Do not copy a record to
a different app path. Partial installations and symlink/junction paths are not reused.
An existing checkout-local SDK DLL is likewise reused and load-tested, not
silently replaced. The runtime doctor checks structure/version, not GPU support.

When no compatible runtime exists, installation requires explicit NVIDIA license
acceptance. The pinned HTTPS archive is approximately 10.6 GB; at least 35 GB free
space is required for the archive and extraction. Interrupted downloads are
retained for resume. ZIP extraction uses the bundled .NET helper rather than
Windows `tar.exe`, preserving Unicode names and supporting paths longer than
260 characters without changing system settings. Entry paths are checked to
reject traversal and links; existing extraction directories are never overwritten.
Failed extraction is retained for diagnosis. Relaunching uses the verified
download again and extracts into a new staging directory, without redownloading.
No existing runtime directory is overwritten, and no post-install global
registration is run.

During extraction, a progress bar updates approximately once per second with
the expanded-byte percentage, GiB processed, completed files, elapsed time and
an estimated remaining time. Plain-text updates are also printed every five
seconds for consoles or logs that do not display PowerShell progress bars.
The estimate is approximate, particularly while extracting many small files.

The SDK is downloaded to the private folder only if no reusable DLL exists.
The launcher passes its exact path to this app rather than registering it globally.
Verified archives remain private until removal; budget disk space accordingly.

```powershell
.\Start_Demo.bat -PrepareOnly
.\Start_Demo.bat -IsaacRoot D:\Tools\isaacsim-6.0.1
.\Start_Demo.bat -InstallBase D:\DemoDependencies
```

`-InstallBase` applies when creating the first installation record, not as an
implicit migration of an existing one. Subsequent launches reuse that record.
An explicit invalid `-IsaacRoot` fails instead of silently installing elsewhere.
An exclusive checkout lock prevents simultaneous start/setup/removal operations.

Approved SDK versions and platform-specific SHA-256 checksums are configured in
`tools/sdk_releases.json`; the binaries come from the FindSurfaceOmniverseExtension
GitHub releases. A recognized older SDK is replaced only after confirmation and
successful download/hash/API checks. A declined or failed update leaves the old
binary in place. An approved local SDK can be used for offline setup; see
[NATIVE_SDK.md](NATIVE_SDK.md).

## Remove dependencies

Close all Kit/Isaac Sim apps, then run `Remove_Demo.bat`. It displays the exact
private directory and preserved external dependencies. Type `REMOVE` only after
checking the list; otherwise nothing is deleted. Removal checks the installation
ID, checkout identity, owner marker and absence of symlinks/junctions before
deleting only the private folder. The deletion is permanent, not a recycle-bin
operation. Do not store personal data in that dependency folder.

The app folder, Git changes, exports, user settings and shared NVIDIA/Packman
caches are not removed. After dependency removal, delete the app folder yourself
only after saving desired files. A later startup can create a new installation.
No ownership record means no automatic removal of legacy dependencies.

This app-specific launcher does not control all NVIDIA caches/settings. They may
remain in NVIDIA-managed locations even when the runtime is private. They are
deliberately outside the remover's scope; do not delete entire NVIDIA/CurvSurf
folders that may be shared with other applications.

## Updates and moved folders

Existing clone users can pull changes and reuse their compatible installations
without reinstalling; previously existing dependencies remain external. ZIP users
should extract new versions separately. Prefer a new private install over reusing
a runtime inside an old ZIP folder you intend to delete. Each app folder has its
own record/private directory; another new app version does not automatically adopt
or delete the old version's private directory.

Keep the old app and record until its remover has been run if cleanup is desired.
After moving an app folder, removal refuses the mismatched checkout identity.
Either move the folder back and run removal, or inspect the old record/owner marker
and manually clean the exact private directory after ensuring it is unused. Rename
the old record for preservation before running setup afresh; do not edit records
to grant ownership of arbitrary existing directories.

Private paths are not advertised globally, but other code can still explicitly
reference them. Ensure no such code depends on them before removal.
