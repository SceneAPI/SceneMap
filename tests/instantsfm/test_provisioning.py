from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scenemap.instantsfm import provisioning


def test_torch_runtime_reinstalls_when_installed_wheel_is_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    steps: list[dict[str, object]] = []
    calls: list[list[str]] = []
    states = iter([("2.12.0+cpu", "cpu"), ("2.11.0+cu128", "12.8")])

    monkeypatch.setattr(provisioning, "_torch_state", lambda: next(states))

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert kwargs["stdin"] is subprocess.DEVNULL
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    provisioning._ensure_torch_runtime(steps, force=False)

    assert steps == [
        {
            "name": "torch_runtime",
            "action": (
                "uv pip install --reinstall --index-url "
                "https://download.pytorch.org/whl/cu128 torch torchvision torchaudio"
            ),
            "status": "done",
            "device": "cuda",
            "torch_version": "2.11.0+cu128",
            "torch_cuda": "12.8",
        }
    ]
    assert calls == [
        [
            "uv",
            "pip",
            "install",
            "--reinstall",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
            "torch",
            "torchvision",
            "torchaudio",
        ]
    ]


def test_dry_run_plans_submodule_and_source_install() -> None:
    result = provisioning.provision(dry_run=True)

    step_names = [step["name"] for step in result["steps"]]

    assert "plugin_source" in step_names
    assert "instantsfm_submodule" in step_names
    assert "instantsfm_source_install" in step_names
    assert "instantsfm_core_dependencies" in step_names
    assert "instantsfm_bae_dependency" in step_names
    assert "instantsfm_gs_dependencies" in step_names


def test_source_root_uses_local_submodule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "plugin"
    root = repo / "third_party" / "instantsfm"
    (root / "instantsfm").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='instantsfm'\n", encoding="utf-8")
    (repo / ".gitmodules").write_text("[submodule 'third_party/instantsfm']\n", encoding="utf-8")
    steps: list[dict[str, object]] = []

    monkeypatch.delenv("SFMAPI_INSTANTSFM_ROOT", raising=False)
    monkeypatch.setattr(provisioning, "REPO_ROOT", repo)
    monkeypatch.setattr(provisioning, "DEFAULT_INSTANTSFM_ROOT", root)

    assert provisioning._source_root(force=False, steps=steps) == root.resolve()
    assert steps == [
        {
            "name": "instantsfm_submodule",
            "action": "use populated submodule third_party/instantsfm",
            "status": "skipped",
            "root": str(root),
        }
    ]


def test_installs_instantsfm_source_without_upstream_deps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "InstantSfM"
    calls: list[list[str]] = []
    steps: list[dict[str, object]] = []

    monkeypatch.setattr(provisioning, "_installed_from_root", lambda module, root: False)
    monkeypatch.setattr(
        provisioning,
        "_run_checked",
        lambda args, steps, **kwargs: (
            calls.append(args)
            or subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        ),
    )

    provisioning._install_instantsfm_source(root, steps, force=False)

    assert calls == [["uv", "pip", "install", "--no-deps", "-e", str(root)]]


def test_runtime_deps_use_curated_replacements(monkeypatch: pytest.MonkeyPatch) -> None:
    core_calls: list[list[str]] = []
    bae_calls: list[list[str]] = []
    steps: list[dict[str, object]] = []
    warnings: list[str] = []

    monkeypatch.setattr(provisioning, "_core_modules_ready", lambda: False)
    monkeypatch.setattr(provisioning, "_installed", lambda module: False)
    monkeypatch.setattr(provisioning, "_cuda_build_env", lambda steps: {})
    monkeypatch.setenv("SFMAPI_INSTANTSFM_SKIP_GS", "1")

    def fake_run(
        args: list[str],
        steps: list[dict[str, object]],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if kwargs["name"] == "instantsfm_core_dependencies":
            core_calls.append(args)
        if kwargs["name"] == "instantsfm_bae_dependency":
            bae_calls.append(args)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(provisioning, "_run_checked", fake_run)

    provisioning._install_core_dependencies(steps, warnings, force=False)
    provisioning._install_bae_dependency(steps, warnings, force=False)

    core_packages = core_calls[0][3:]
    assert "scipy==1.13.0" in core_packages
    assert "sksparse-minimal>=0.3" not in core_packages
    assert "scikit-sparse==0.4.15" not in core_packages
    assert "gsplat" not in core_packages
    assert all("pypose.git@bae" not in package for package in core_packages)
    assert bae_calls == [["uv", "pip", "install", "--no-build-isolation", "bae-kai"]]


def test_gs_deps_build_without_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    steps: list[dict[str, object]] = []
    warnings: list[str] = []

    monkeypatch.setattr(provisioning, "_installed", lambda module: False)
    monkeypatch.setattr(provisioning, "_cuda_build_env", lambda steps: {"CUDA_HOME": "C:/CUDA"})
    monkeypatch.setenv("SFMAPI_INSTANTSFM_INSTALL_GS", "1")
    monkeypatch.setattr(
        provisioning,
        "_run_checked",
        lambda args, steps, **kwargs: (
            calls.append(args)
            or subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        ),
    )

    provisioning._install_gaussian_splatting_dependencies(steps, warnings, force=False)

    assert calls == [
        [
            "uv",
            "pip",
            "install",
            "--no-build-isolation",
            "gsplat",
            "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim",
        ]
    ]


def test_torch_arch_list_has_container_build_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(provisioning, "_installed", lambda module: True)

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _Torch:
        cuda = _Cuda()

    monkeypatch.setitem(sys.modules, "torch", _Torch())

    assert provisioning._torch_arch_list() == provisioning.DEFAULT_TORCH_CUDA_ARCH_LIST


def test_patch_instantsfm_source_rewrites_colmap_flags(tmp_path: Path) -> None:
    root = tmp_path / "InstantSfM"
    controller = root / "instantsfm" / "controllers" / "feature_handler.py"
    scene_defs = root / "instantsfm" / "scene" / "defs.py"
    rotation_averaging = root / "instantsfm" / "processors" / "rotation_averaging.py"
    controller.parent.mkdir(parents=True)
    scene_defs.parent.mkdir(parents=True)
    rotation_averaging.parent.mkdir(parents=True)
    controller.write_text(
        "\n".join(
            [
                '"--SiftExtraction.use_gpu"',
                '"--SiftMatching.use_gpu"',
                'print(f"Error during feature extraction: {e}")',
                'print(f"Error during exhaustive matching: {e}")',
            ]
        ),
        encoding="utf-8",
    )
    scene_defs.write_text(
        "\n".join(
            [
                "self.ids = np.full(num_tracks, -1, dtype=np.int32)",
                "self.ids = np.zeros(self.num_tracks, dtype=np.int32)",
            ]
        ),
        encoding="utf-8",
    )
    rotation_averaging.write_text(
        "\n".join(
            [
                "        self.images = {image_id: image for image_id, image in enumerate(images) if registered_mask[image_id]}",
                "        self.image_pairs = {pair_key: pair for pair_key, pair in view_graph.image_pairs.items() if pair.is_valid}",
                "        if self.fixed_camera_id == -1:",
            ]
        ),
        encoding="utf-8",
    )
    steps: list[dict[str, object]] = []

    provisioning._patch_instantsfm_source(root, steps)

    patched = controller.read_text(encoding="utf-8")
    assert "--FeatureExtraction.use_gpu" in patched
    assert "--FeatureMatching.use_gpu" in patched
    assert "raise RuntimeError" in patched
    patched_defs = scene_defs.read_text(encoding="utf-8")
    assert "np.full(num_tracks, -1, dtype=np.int64)" in patched_defs
    assert "np.zeros(self.num_tracks, dtype=np.int64)" in patched_defs
    patched_rotation = rotation_averaging.read_text(encoding="utf-8")
    assert "registered_ids = set(self.images)" in patched_rotation
    assert "and pair.image_id1 in registered_ids" in patched_rotation
    assert "if self.fixed_camera_id not in self.image_id2idx:" in patched_rotation
    assert steps[-1]["status"] == "done"
