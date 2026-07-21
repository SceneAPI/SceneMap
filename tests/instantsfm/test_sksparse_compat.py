from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import numpy as np
from scipy.sparse import csc_matrix

from scenemap.instantsfm.backend import SKSPARSE_SHIM_DIR, instantsfm_pythonpath


def test_shim_is_not_installed_as_top_level_sksparse() -> None:
    """The shim is plugin-private: installing scenemap must never own
    a top-level ``sksparse`` package that shadows real scikit-sparse."""
    spec = importlib.util.find_spec("sksparse")

    if spec is not None and spec.origin:
        assert "scenemap.instantsfm" not in spec.origin


def test_worker_pythonpath_prepends_root_and_shim(tmp_path) -> None:
    value = instantsfm_pythonpath(tmp_path, existing="already")

    parts = value.split(os.pathsep)
    assert parts[0] == str(tmp_path)
    assert parts[1] == str(SKSPARSE_SHIM_DIR)
    assert parts[-1] == "already"
    assert (SKSPARSE_SHIM_DIR / "sksparse" / "cholmod.py").is_file()
    # Idempotent: re-deriving from an already-injected value adds nothing.
    assert instantsfm_pythonpath(tmp_path, existing=value) == value


def test_sksparse_cholmod_compat_factor_solves_linear_system() -> None:
    sys.path.insert(0, str(SKSPARSE_SHIM_DIR))
    try:
        from sksparse.cholmod import cholesky

        dense = np.array([[4.0, 1.0], [1.0, 3.0]])
        matrix = csc_matrix(dense)
        rhs = np.array([1.0, 2.0])

        factor = cholesky(matrix)

        assert np.allclose(factor(rhs), np.linalg.solve(dense, rhs))
    finally:
        sys.path.remove(str(SKSPARSE_SHIM_DIR))
        sys.modules.pop("sksparse.cholmod", None)
        sys.modules.pop("sksparse", None)


def test_worker_subprocess_resolves_sksparse_via_injected_pythonpath(tmp_path) -> None:
    """End-to-end: a worker subprocess (the way ``_run_python_module`` spawns
    the engine) resolves ``sksparse.cholmod`` from the injected PYTHONPATH."""
    env = os.environ.copy()
    env["PYTHONPATH"] = instantsfm_pythonpath(tmp_path, env.get("PYTHONPATH"))

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sksparse.cholmod, pathlib, sys;"
            "sys.stdout.write(str(pathlib.Path(sksparse.cholmod.__file__)))",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert str(SKSPARSE_SHIM_DIR) in completed.stdout
