# scenemap

One package for the four SfM mapping plugin families that previously shipped
as four separate repos (`sfmapi_colmap_unified`, `sfmapi_instantsfm`,
`sfmapi_spheresfm`, `sfmapi_realityscan`). Each plugin keeps its own manifest,
entry point, and launcher; the COLMAP family keeps the internal structure it
already had from its own three-repo merge, re-rooted under
`scenemap.colmap`. This is the first release against the renamed core:
the dependency is `sceneapi` (0.1.x), imports use the `sceneapi.*` facades,
and the entry-point group is `sceneapi.backends`.

| Entry point (`sceneapi.backends`) | Plugin module | Provider ids | Launcher |
|---|---|---|---|
| `colmap_native` | `scenemap.colmap.native.plugin` | `colmap_cli`, `colmap_pycolmap`, `colmap_cpp_native`, `colmap_cpp_inmemory` | `sfmapi-colmap-api` |
| `pycolmap` | `scenemap.colmap.pycolmap.plugin` | `colmap_pycolmap` | `sfmapi-pycolmap-api` |
| `colmap_cli` | `scenemap.colmap.cli.plugin` | `colmap_cli` | `sfmapi-colmap-cli-api` |
| `instantsfm` | `scenemap.instantsfm.plugin` | `instantsfm` | `sfmapi-instantsfm-api` |
| `spheresfm` | `scenemap.spheresfm.plugin` | `spheresfm` | `sfmapi-spheresfm-api` |
| `realityscan_cli` | `scenemap.realityscan.plugin` | `realityscan_cli` | `sfmapi-realityscan-api` |

Family notes:

- **COLMAP** — the three `backend.py` adapter implementations (`native/`,
  `pycolmap/`, `cli/`) are genuinely distinct and are carried as-is per
  provider; the shared modules (`model.py`, `provisioning.py`,
  `api_launcher.py`, the reconciled `pycolmap_backend.py`) exist once.
- **InstantSfM** — action-first wrapper over the upstream global-SfM engine
  (portable `map.global` via a path-staging adapter). Ships the
  plugin-private SciPy-backed `sksparse.cholmod` shim
  (`scenemap/instantsfm/_sksparse_shim/`, PYTHONPATH-injected into worker
  subprocesses — deliberately never installed as a top-level `sksparse`
  package). Also exposes an `sfmapi-plugin-http-v1` container service
  (`sfmapi-instantsfm-service`, `Dockerfile`).
- **SphereSfM** — COLMAP-fork spherical SfM; full sparse pipeline as portable
  capabilities plus `spheresfm.*` actions, driven through the SphereSfM
  `colmap` executable you build from the submodule.
- **RealityScan** — action-only wrapper over the proprietary RealityScan /
  RealityCapture CLI (Windows); portable capability set is deliberately
  empty.

## Install and extras

```bash
uv pip install scenemap              # CLI-driven providers work with engines on PATH
uv pip install "scenemap[pycolmap]"  # + pycolmap wheel for the in-process provider
```

Extras (union of the four source repos, deduped; layouts unchanged):

- `pycolmap` — `pycolmap>=4.0.4` (the in-process COLMAP engine).
- `pycolmap-cuda12` — CUDA 12 pycolmap wheel (Linux).
- `mcp` — `fastmcp` for the `--mcp local` launcher mode.
- `standalone` — Nuitka + zstandard + fastmcp for the standalone API build
  (`scripts/build-nuitka-standalone.*`, COLMAP native provider).
- `dev` / `test` — pytest, ruff (+ httpx for the HTTP tests).

`scipy` is a main dependency (it backs InstantSfM's `sksparse.cholmod` shim,
exactly as in `sfmapi-instantsfm`).

## Running a provider API

The launchers keep their old names, flags, and defaults:

```bash
sfmapi-colmap-api --backend colmap_cpp_native      # native collection (default unchanged)
sfmapi-pycolmap-api --backend colmap_pycolmap      # pycolmap (default)
sfmapi-colmap-cli-api --colmap-executable <path>   # pinned to colmap_cli
sfmapi-instantsfm-api --instantsfm-root <checkout>
sfmapi-spheresfm-api --spheresfm-executable <path>
sfmapi-realityscan-api --rc-executable <path>
```

The `sfmapi-*-info` commands print each provider's runtime versions as
before, and `sfmapi-instantsfm-service` runs the InstantSfM
`sfmapi-plugin-http-v1` container service. ASGI module paths for uvicorn
import strings: `scenemap.colmap.{native,pycolmap,cli}.server:app`,
`scenemap.instantsfm.server:app`, `scenemap.spheresfm.server:app`,
`scenemap.realityscan.server:app`. Example launch scripts live in
`examples/` (one per family).

## Development

```bash
uv venv
uv sync --extra dev
uv run ruff check .
uv run pytest -q
```

`sceneapi` resolves from the sibling checkout (`../sfmapi` — the core repo's
directory keeps its old name on disk) via `[tool.uv.sources]`; CI checks out
`SceneAPI/SceneAPI` into `.deps/sceneapi` and installs this package
`--no-deps`.

Tests are namespaced per family (`tests/colmap/`, `tests/instantsfm/`,
`tests/spheresfm/`, `tests/realityscan/`); the four byte-identical
`test_public_boundary.py` copies were deduped into one shared
`tests/test_public_boundary.py`. Engine-gated tests keep their
`needs_colmap` / `needs_pycolmap` / `needs_sample_data` / `gpu` / `slow` /
`integration` / `e2e` markers; CI runs the lane that excludes all of those.

### third_party submodules

Three upstream engines are vendored as git submodules, registered without
their contents checked out (CI never initializes them; every test that
touches them skips while uninitialized). Initialize only what you need:

```bash
git submodule update --init third_party/colmap       # COLMAP source builds (scripts/build-colmap.*)
git submodule update --init third_party/instantsfm   # InstantSfM engine (provisioning installs from it)
git submodule update --init third_party/spheresfm    # SphereSfM source (build its colmap executable)
```

Same URLs and pinned commits as the source repos: colmap/colmap
`6cfbc0404cb973ca79a6835bb88bb020ed96bc8f`, cre185/InstantSfM
`e2b357a0d359e29b84e941fecd345fd5348c8578`, json87/SphereSfM
`6b40b2d3ca56dd114700513856c9490283959990`. Nothing at runtime requires the
submodules; provisioning and the build scripts materialize or clone what
they need. Known pre-existing issue carried from `sfmapi_instantsfm`: after
initializing `third_party/instantsfm`,
`tests/instantsfm/test_instantsfm_upstream_patches.py` asserts the upstream
int64 track-id patch is present in the working tree — run the InstantSfM
provisioner (which applies `_patch_instantsfm_source`) if it fails on a
stale checkout; with the submodule uninitialized the test skips.

### Native C++ demo extension (not built here)

The superseded `sfmapi_colmap` repo built a small pybind11 demo extension
(`sfmapi_colmap._cpp_inmemory`, one `.cpp`, scikit-build-core) backing the
`colmap_cpp_native` / `colmap_cpp_inmemory` providers. That build is
**deliberately not wired into this package** (hatchling, pure Python): both
C++ providers degrade gracefully without it (import fine, register fine,
`capabilities() == set()`, raise `CapabilityUnavailableError` with guidance
when driven), and the backends keep looking up `sfmapi_colmap._cpp_inmemory`
from a wheel built out of the superseded repo. Tests that drive the real
extension skip when it is not installed.

## Migration (SceneAPI proposal W8 — supersedes four repos)

This package supersedes `sfmapi_colmap_unified`, `sfmapi_instantsfm`,
`sfmapi_spheresfm`, and `sfmapi_realityscan`, which are to be archived by
their owner after verification (`sfmapi_colmap_unified` had already
superseded `sfmapi_colmap`, `sfmapi_pycolmap`, and `sfmapi_colmap_cli`).

- **Entry-point names are unchanged** — `colmap_native`, `pycolmap`,
  `colmap_cli`, `instantsfm`, `spheresfm`, `realityscan_cli` — but they now
  live in the `sceneapi.backends` group (the renamed core also reads the
  legacy `sfmapi.backends` group for one release). Only the entry-point
  *values* move (e.g. `sfmapi_instantsfm.plugin:plugin` →
  `scenemap.instantsfm.plugin:plugin`).
- **Console scripts are unchanged**: all thirteen `sfmapi-*` scripts from
  the four source repos are preserved byte-for-name.
- **Manifest identity now names this repo**: `package_name`
  (`scenemap`), `github_url`
  (`https://github.com/SceneAPI/SceneMap.git`), `entry_points`
  (`scenemap.<family>...:plugin`), the `uv` runtime-mode install
  coordinates, and InstantSfM's `container_service.image.build.context`
  (its `Dockerfile` moved here, still at the repo root). Plugin ids,
  provider ids, capability/action/config/artifact declarations,
  `external_tool` executable/env names, engine upstream coordinates, and
  the `compatibility.sfmapi` field are untouched, so hub state keyed on
  those keeps working. The core's bundled `sfm_hub` registry still carries
  the pre-merge coordinates until the W9 sweep; the registry-comparison
  test is skip-gated until then.
- **Env vars are unchanged**: plugins keep reading/writing `SFMAPI_*`
  names (`SFMAPI_COLMAP_EXECUTABLE`, `SFMAPI_INSTANTSFM_ROOT`, ...); the
  renamed core honors the `SFMAPI_*` prefix via a one-release alias.
- **Direct importers**: `sfmapi_colmap_unified.X` → `scenemap.colmap.X`;
  `sfmapi_instantsfm.X` → `scenemap.instantsfm.X`; `sfmapi_spheresfm.X`
  → `scenemap.spheresfm.X`; `sfmapi_realityscan.X` →
  `scenemap.realityscan.X` (the colmap subpackage's own mapping from
  the D3/L43 three-repo merge is reproduced below).

## RECONCILIATION NOTES (ported from the source merges)

### `sfmapi_colmap_unified` (now `scenemap.colmap`) — D3 / register L43

Carried unchanged from that repo's merge of `sfmapi_colmap`,
`sfmapi_pycolmap`, and `sfmapi_colmap_cli`:

- **`pycolmap_backend.py` reconciliation.** The module existed in both
  `sfmapi_colmap` (921 lines, last touched 2026-05-14) and
  `sfmapi_pycolmap` (1,765 lines, 2026-05-27) — same purpose, silently
  diverged. The base is `sfmapi_pycolmap`'s version (the dedicated repo;
  strictly newer — it carries the 2026-05-27 canonical ordered COLMAP
  `pair_id` fix the fork never received). The fork was diffed hunk-by-hunk;
  **no fork delta was ported** — each was an older or narrower strategy for
  something the base does in-process:
  1. fork's `run_mapping` pose-prior delegation to the COLMAP CLI
     `pose_prior_mapper` — rejected; the base writes priors into the
     database's `pose_priors` table and enables `use_prior_position`
     in-process (the CLI path lives on in `native/backend.py`).
  2. fork's `estimate_two_view_geometry` via pairs-file +
     `pycolmap.verify_matches` — rejected; the base estimates E/F/H +
     relative pose directly with the `pycolmap.estimate_*` bindings (the
     contract distinguishes this stage from `verify_matches`).
  3. fork's CLI-gated `capabilities()` for `georegister.gps` /
     `pose_priors.mapping` and missing `localize.from_memory` — rejected;
     the base implements them in-process, so they are advertised always-on.
  4. cosmetic import/ordering differences — rejected.
  Base-only surfaces (all kept): `localize_from_memory`,
  `align_reconstruction`, `undistort_images`, `build_vocab_tree`,
  `configure_rig`, `pose_graph_optimize`, the in-process pose-prior wiring,
  the two-view helper suite, and the canonical-`pair_id` ordering fix.
  Consequence: the `colmap_native` plugin's `colmap_pycolmap` provider uses
  the reconciled backend; its manifest capability list still understates it
  (e.g. `localize.from_memory`) exactly as far as the bundled registry copy
  does, until manifest identity is re-pointed in the W9 registry sweep.
- **`provisioning.py`** exists once; per-provider entry points
  (`provision_native` / `provision_pycolmap` / `provision_cli`) preserve
  each source repo's exact step set and detection strategy, and the
  package-level hook `provision()` runs the superset (= native) path. The
  GitHub API User-Agent follows the package name (now `scenemap`;
  cosmetic).
- **`api_launcher.py`** is one provider-parameterized module. Preserved per
  provider: prog names, `--backend` choices and defaults, the cli launcher
  force-pinning `SFMAPI_BACKEND=colmap_cli`, native's bundled-COLMAP
  detection for frozen builds, and each provider's COLMAP-executable
  resolution strategy. Deliberate deltas (from that merge): `--dry-run` and
  `--workspace-root` exist for every provider, `--colmap` is accepted as an
  alias everywhere, and the non-reload path passes uvicorn the import
  string.
- **`model.py`** single copy — the three sources were md5-identical.
- **Tests** are ported per provider (`test_native_*`, `test_pycolmap_*`,
  `test_cli_*`); byte-identical or import-only-diff copies were deduped
  into shared/parametrized tests. Engine-gated tests keep their skip
  conditions; the merge-specific skips (C++ demo extension not installed;
  submodule not initialized) are carried.
- The per-repo `scripts/build-colmap.*` divergence is preserved as
  `build-colmap.*` (full executable build) + `build-pycolmap.*`
  (pycolmap-from-submodule); the cli repo's minimal wrapper stayed dropped.

### This merge (W8) — reconciliation and adaptations

The four families shared no Python modules, so no module-level
reconciliation was needed; everything family-specific is carried as-is under
its subpackage. The deliberate deltas, all mechanical:

- The four byte-identical `test_public_boundary.py` copies became one
  shared test; suites are namespaced per family (same filenames kept).
- `REPO_ROOT` computations gained one `parents[]` level (package moved one
  directory deeper); colmap's gained one more on top of its own merge bump.
- The three per-family `examples/run-server.ps1` were renamed
  `run-server-{instantsfm,spheresfm,realityscan}.ps1` (name collision).
- The http-discovery tests of the three single-plugin families pin
  `auto_load_backend_plugins=False`, the same adaptation the colmap suite
  already carried ("the unified package installs all N entry points").
- `test_cli_plugin_contract`'s fake entry-point source now answers the
  renamed core's dual group read (`sceneapi.backends` + legacy).
- The Nuitka standalone scripts bundle the core as `sceneapi` (the `app`
  shim they previously named was removed in core 0.1.0).
- InstantSfM's `Dockerfile` installs the core from
  `SceneAPI/SceneAPI@SCENEAPI_REF` and this package as `scenemap`;
  engine build steps are untouched.
- InstantSfM's upstream-patch test gained the standard
  submodule-uninitialized skip gate (its source repo always ran with the
  submodule checked out).
