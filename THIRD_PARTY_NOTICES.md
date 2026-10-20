# Third-Party Notices

This application integrates with NVIDIA Isaac Sim, Omniverse Kit and OpenUSD.
Their runtime components are obtained separately under their respective terms.
NVIDIA product names are trademarks of NVIDIA Corporation; this application is
not endorsed by NVIDIA unless stated otherwise in writing.

The FindSurface SDK is owned by CurvSurf, Inc. It is obtained separately and is
not included in this repository. **Its native binaries, including
`FindSurface.dll` and `libFindSurface.so`, are expressly excluded from the
application's MIT license.** Their use and redistribution follow the SDK's
release terms and any applicable commercial agreement, even when the launch
scripts download them automatically.

The reusable FindSurface extension is maintained by CurvSurf:
https://github.com/CurvSurf/FindSurfaceOmniverseExtension
Its accompanying THIRD_PARTY_NOTICES.md is retained in the submodule. The
application's MIT license does not relicense that separate repository.

## Tabler Icons

Source: https://github.com/tabler/tabler-icons

Tabler-derived toolbar and manual icons are distributed under the MIT License.
The full license is retained at
`source/extensions/com.curvsurf.findsurface_isaac_sim/data/icons/tabler/LICENSE`.
Adaptations include application colors, composed camera-reset symbols, and dark
background tiles for the manual. FindSurface-specific artwork is supplied by
CurvSurf.

## Application source

The CurvSurf-owned application code and documentation are licensed under the
[MIT License](LICENSE), copyright (c) 2026 CurvSurf, Inc. This covers the
application's native point-picking helper, which is distinct from the
FindSurface SDK binaries. Third-party components and the separate extension
submodule retain their own applicable terms and notices.
