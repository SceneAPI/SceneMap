# Third-Party Notices

This repository contains the unified sceneapi SfM mapping backend wrappers
(COLMAP native/PyCOLMAP/CLI, InstantSfM, SphereSfM, and RealityScan
providers). The wrapper source code in this repository is licensed under the
Apache License, Version 2.0. See `LICENSE`.

## COLMAP and PyCOLMAP

The upstream COLMAP source is included as a git submodule under
`third_party/colmap`. PyCOLMAP exposes Python bindings for upstream
COLMAP; this repository may use PyCOLMAP from a separately installed
wheel or from the `third_party/colmap` submodule when built locally.
COLMAP itself is licensed under the new BSD license. A copy of COLMAP's
original license text is included at
`LICENSES/COLMAP-BSD-3-Clause.txt` and remains available in the
submodule at `third_party/colmap/COPYING.txt` when the submodule is
initialized.

COLMAP and PyCOLMAP dependencies are separately licensed. Building or
redistributing native wheels or binaries may require including
additional notices from the dependency set used for that build.

## InstantSfM

The upstream InstantSfM project is included as a git submodule at
`third_party/instantsfm`.

- Repository: https://github.com/cre185/InstantSfM
- License: Creative Commons Attribution-NonCommercial 4.0 International
- Copied notice: `LICENSES/InstantSfM-CC-BY-NC-4.0.txt`

This wrapper does not relicense InstantSfM. That non-commercial term is
upstream's and binds whoever runs InstantSfM; review the upstream license
before commercial use.

## SphereSfM

The upstream SphereSfM project is included as a git submodule at
`third_party/spheresfm`.

- Repository: https://github.com/json87/SphereSfM
- License: BSD-3-Clause
- Copied notice: `LICENSES/SphereSfM-BSD-3-Clause.txt`

SphereSfM is derived from COLMAP and includes additional third-party code in
its upstream tree. Review `third_party/spheresfm/COPYING.txt` and upstream
`lib/*/LICENSE` files when distributing builds.

## RealityCapture / RealityScan

RealityCapture and RealityScan are proprietary Epic Games / Capturing Reality
applications. This repository does not vendor, copy, or redistribute those
executables, their models, their assets, or their bundled third-party
dependencies.

Users must install RealityCapture or RealityScan separately and use it under
the original vendor license terms that apply to their installed version. If a
redistribution package ever bundles a RealityCapture or RealityScan
executable, that package must include the original vendor license/EULA and
any notices from the application's installed `Licenses/` directory.

The backend only invokes the installed executable through its documented
command line interface.
