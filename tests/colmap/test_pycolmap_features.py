from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sceneapi_map.colmap.model import Reconstruction
from sceneapi_map.colmap.pycolmap.backend import (
    COLMAP_COMMANDS,
    COLMAP_EXPORT_TYPES,
    CapabilityUnavailableError,
    ColmapCliBackend,
    ValidationError,
)


def _capture_colmap(monkeypatch: pytest.MonkeyPatch, backend: ColmapCliBackend):
    calls: list[list[str]] = []
    required: list[str] = []

    def require_colmap(capability: str) -> str:
        required.append(capability)
        return "colmap"

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        normalized = [str(arg) for arg in args]
        calls.append(normalized)
        return subprocess.CompletedProcess(normalized, 0, "stdout", "stderr")

    monkeypatch.setattr(backend, "_require_colmap", require_colmap)
    monkeypatch.setattr(backend, "_run", run)
    return calls, required


def _write_text_model(path: Path, image_name: str = "image.jpg") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "cameras.txt").write_text(
        "1 SIMPLE_PINHOLE 640 480 500 320 240\n",
        encoding="utf-8",
    )
    (path / "images.txt").write_text(
        f"1 1 0 0 0 0 0 0 1 {image_name}\n10 20 -1\n",
        encoding="utf-8",
    )
    (path / "points3D.txt").write_text("", encoding="utf-8")


def _sample_reconstruction(num_images: int = 2, num_points: int = 3) -> Reconstruction:
    return Reconstruction(
        images={idx: object() for idx in range(1, num_images + 1)},
        points3D={idx: object() for idx in range(1, num_points + 1)},
    )


def _value_after(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


def test_capabilities_are_empty_when_colmap_is_missing(monkeypatch: pytest.MonkeyPatch):
    backend = ColmapCliBackend()
    monkeypatch.setattr(backend, "_find_colmap", lambda: None)

    assert backend.capabilities() == set()


def test_capabilities_cover_all_advertised_feature_paths(tmp_path: Path):
    fake_colmap = tmp_path / "colmap"
    fake_colmap.write_text("", encoding="utf-8")

    assert ColmapCliBackend(executable=fake_colmap).capabilities() == {
        "features.extract.sift",
        "matches.verify",
        "pairs.exhaustive",
        "pairs.sequential",
        "pairs.spatial",
        "pairs.vocabtree",
        "pairs.explicit",
        "matchers.nn-mutual",
        "matchers.nn-ratio",
        "map.incremental",
        "map.global",
        "map.hierarchical",
        "ba.standard",
        "triangulate.retri",
        "relocalize.images",
        "recon.merge",
        "export.ply",
        "export.nvm",
        "export.colmap_text",
        "export.colmap_bin",
        "georegister.sim3",
    }


def test_extract_features_writes_image_list_and_builds_feature_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    image_list_payloads: list[str] = []

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        normalized = [str(arg) for arg in args]
        list_path = Path(_value_after(normalized, "--image_list_path"))
        image_list_payloads.append(list_path.read_text(encoding="utf-8"))
        calls.append(normalized)
        return subprocess.CompletedProcess(normalized, 0, "stdout", "stderr")

    monkeypatch.setattr(backend, "_run", run)

    result = backend.extract_features(
        database_path=tmp_path / "db" / "database.db",
        image_root=tmp_path / "images",
        image_list=["a.jpg", "nested/b.jpg"],
        options={"sift": {"max_num_features": 2048, "use_gpu": False}},
    )

    args = calls[0]
    assert required == ["features.extract.sift"]
    assert args[1] == "feature_extractor"
    assert _value_after(args, "--database_path").endswith("database.db")
    assert _value_after(args, "--image_path").endswith("images")
    assert image_list_payloads == ["a.jpg\nnested/b.jpg\n"]
    assert "--SiftExtraction.max_num_features" in args
    assert _value_after(args, "--FeatureExtraction.use_gpu") == "0"
    assert not Path(_value_after(args, "--image_list_path")).exists()
    assert result["num_images"] == 2


@pytest.mark.parametrize(
    ("mode", "command", "prefix"),
    [
        ("exhaustive", "exhaustive_matcher", "ExhaustiveMatching"),
        ("sequential", "sequential_matcher", "SequentialMatching"),
        ("spatial", "spatial_matcher", "SpatialMatching"),
        ("vocab-tree", "vocab_tree_matcher", "VocabTreeMatching"),
        ("transitive", "transitive_matcher", "TransitiveMatching"),
    ],
)
def test_match_builds_each_supported_matcher_command(
    mode: str,
    command: str,
    prefix: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)

    result = backend.match(
        database_path=tmp_path / "database.db",
        mode=mode,
        options={"max_num_matches": 500, "mode": "ignored", "nested": {"skip": True}},
    )

    args = calls[0]
    capability_mode = mode.replace("-", "_")
    expected_capability = (
        f"pairs.{capability_mode.replace('_', '')}"
        if capability_mode in {"exhaustive", "sequential", "spatial", "vocab_tree"}
        else "backend.actions"
    )
    assert required == [expected_capability]
    assert args[:2] == ["colmap", command]
    assert _value_after(args, "--database_path").endswith("database.db")
    assert _value_after(args, "--FeatureMatching.max_num_matches") == "500"
    assert result["engine"] == f"colmap {command}"


def test_match_rejects_unknown_strategy():
    with pytest.raises(CapabilityUnavailableError):
        ColmapCliBackend().match(
            database_path=Path("database.db"), mode="gpu-brute-force", options={}
        )


def test_verify_matches_runs_geometric_verifier_and_counts_verified_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(
        backend,
        "iter_two_view_geometries",
        lambda *, database_path: iter([(1, 2, object()), (2, 3, object())]),
    )

    result = backend.verify_matches(
        database_path=tmp_path / "database.db",
        options={"max_error": 3.5},
    )

    args = calls[0]
    assert required == ["matches.verify"]
    assert args[:2] == ["colmap", "geometric_verifier"]
    assert _value_after(args, "--TwoViewGeometry.max_error") == "3.5"
    assert result["num_verified_pairs"] == 2


@pytest.mark.parametrize(
    ("kind", "command"),
    [("incremental", "mapper"), ("hierarchical", "hierarchical_mapper")],
)
def test_run_mapping_builds_mapping_command_and_converts_models(
    kind: str,
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    sparse_root = tmp_path / "sparse"
    model_dir = sparse_root / "0"
    model_dir.mkdir(parents=True)
    conversions: list[tuple[Path, Path, str]] = []

    def convert_model(input_path: Path, output_path: Path, output_type: str) -> None:
        conversions.append((input_path, output_path, output_type))
        _write_text_model(output_path)

    monkeypatch.setattr(backend, "_convert_model", convert_model)

    summaries, reconstructions = backend.run_mapping(
        kind=kind,
        db_path=tmp_path / "database.db",
        image_root=tmp_path / "images",
        sparse_root=sparse_root,
        job_dir=tmp_path / "job",
        spec={"mapper": {"ba_refine_focal_length": True}},
    )

    args = calls[0]
    assert required == [f"map.{kind}"]
    assert args[:2] == ["colmap", command]
    assert _value_after(args, "--output_path") == str(sparse_root)
    assert _value_after(args, "--Mapper.ba_refine_focal_length") == "1"
    assert conversions == [(model_dir, tmp_path / "job" / "colmap_text_models" / "0", "TXT")]
    assert summaries == [
        {"idx": 0, "num_reg_images": 1, "num_points3D": 0, "model_path": str(model_dir)}
    ]
    assert len(reconstructions) == 1


def test_run_mapping_rejects_unknown_mapper_kind():
    with pytest.raises(CapabilityUnavailableError):
        ColmapCliBackend().run_mapping(
            kind="dense",
            db_path=Path("database.db"),
            image_root=Path("images"),
            sparse_root=Path("sparse"),
            job_dir=Path("job"),
            spec={},
        )


def test_bundle_adjustment_builds_command_and_summarizes_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(backend, "read_reconstruction", lambda path: _sample_reconstruction())

    result = backend.bundle_adjustment(
        model_path=tmp_path / "model",
        output_path=tmp_path / "ba",
        spec={"mode": "final", "bundle_adjustment": {"refine_focal_length": False}},
    )

    args = calls[0]
    assert required == ["ba.standard"]
    assert args[:2] == ["colmap", "bundle_adjuster"]
    assert _value_after(args, "--output_path").endswith("ba")
    assert _value_after(args, "--BundleAdjustment.refine_focal_length") == "0"
    assert result["mode"] == "final"
    assert result["num_reg_images"] == 2
    assert result["num_points3D"] == 3


def test_triangulate_builds_point_triangulator_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(backend, "read_reconstruction", lambda path: _sample_reconstruction(1, 4))

    result = backend.triangulate(
        model_path=tmp_path / "model",
        database_path=tmp_path / "database.db",
        image_root=tmp_path / "images",
        output_path=tmp_path / "triangulated",
    )

    args = calls[0]
    assert required == ["triangulate.retri"]
    assert args[:2] == ["colmap", "point_triangulator"]
    assert _value_after(args, "--image_path").endswith("images")
    assert result["num_points3D"] == 4


def test_relocalize_builds_image_registrator_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(backend, "read_reconstruction", lambda path: _sample_reconstruction(3, 1))

    result = backend.relocalize(
        model_path=tmp_path / "model",
        database_path=tmp_path / "database.db",
        image_root=tmp_path / "images",
        output_path=tmp_path / "registered",
        image_ids=[8, 9],
    )

    args = calls[0]
    assert required == ["relocalize.images"]
    assert args[:2] == ["colmap", "image_registrator"]
    assert "--image_path" not in args
    assert result["requested_image_ids"] == [8, 9]
    assert result["num_reg_images"] == 3


@pytest.mark.parametrize(("format_key", "output_type"), sorted(COLMAP_EXPORT_TYPES.items()))
def test_export_supports_each_model_converter_format(
    format_key: str,
    output_type: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    conversions: list[tuple[Path, Path, str]] = []

    def convert_model(input_path: Path, output_path: Path, output_type_arg: str) -> None:
        conversions.append((input_path, output_path, output_type_arg))

    monkeypatch.setattr(backend, "_convert_model", convert_model)

    result = backend.export(
        model_path=tmp_path / "model",
        output_path=tmp_path / "exports" / format_key,
        format=format_key,
    )

    assert calls == []
    assert required == [f"export.{format_key}"]
    assert conversions == [(tmp_path / "model", tmp_path / "exports" / format_key, output_type)]
    assert result["format"] == format_key


def test_export_normalizes_hyphenated_format_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = ColmapCliBackend()
    _calls, required = _capture_colmap(monkeypatch, backend)
    conversions: list[str] = []
    monkeypatch.setattr(
        backend,
        "_convert_model",
        lambda input_path, output_path, output_type: conversions.append(output_type),
    )

    result = backend.export(
        model_path=tmp_path / "model",
        output_path=tmp_path / "model.txt",
        format="colmap-text",
    )

    assert required == ["export.colmap_text"]
    assert conversions == ["TXT"]
    assert result["format"] == "colmap_text"


def test_export_rejects_unknown_format(tmp_path: Path):
    with pytest.raises(CapabilityUnavailableError):
        ColmapCliBackend().export(
            model_path=tmp_path / "model",
            output_path=tmp_path / "out",
            format="las",
        )


def test_merge_reconstructions_builds_model_merger_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(backend, "read_reconstruction", lambda path: _sample_reconstruction(4, 5))

    result = backend.merge_reconstructions(
        model_paths=[tmp_path / "a", tmp_path / "b"],
        output_path=tmp_path / "merged",
    )

    args = calls[0]
    assert required == ["recon.merge"]
    assert len(calls) == 1
    assert args[:2] == ["colmap", "model_merger"]
    assert _value_after(args, "--input_path1").endswith("a")
    assert _value_after(args, "--input_path2").endswith("b")
    # Two-model merge writes straight into the output path.
    assert _value_after(args, "--output_path") == str(tmp_path / "merged")
    assert result["num_reg_images"] == 4
    assert result["num_points3D"] == 5
    assert result["num_sources"] == 2


def test_merge_reconstructions_folds_n_models_pairwise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(backend, "read_reconstruction", lambda path: _sample_reconstruction(7, 9))

    result = backend.merge_reconstructions(
        model_paths=[tmp_path / "a", tmp_path / "b", tmp_path / "c"],
        output_path=tmp_path / "merged",
        sim3_aligners={"ignored": True},
    )

    # COLMAP model_merger is pairwise: 3 models -> 2 fold-left calls.
    assert required == ["recon.merge"]
    assert len(calls) == 2
    first, second = calls
    assert _value_after(first, "--input_path1").endswith("a")
    assert _value_after(first, "--input_path2").endswith("b")
    # The first fold feeds the second fold's input_path1.
    assert _value_after(second, "--input_path1") == _value_after(first, "--output_path")
    assert _value_after(second, "--input_path2").endswith("c")
    # The final fold writes into the requested output path.
    assert _value_after(second, "--output_path") == str(tmp_path / "merged")
    assert result["num_sources"] == 3


def test_merge_reconstructions_requires_two_models(tmp_path: Path):
    with pytest.raises(ValidationError):
        ColmapCliBackend().merge_reconstructions(
            model_paths=[tmp_path / "only"],
            output_path=tmp_path / "merged",
        )


def test_apply_sim3_builds_model_transformer_command_and_temp_transform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(backend, "read_reconstruction", lambda path: _sample_reconstruction(1, 2))
    transform_payloads: list[str] = []

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        normalized = [str(arg) for arg in args]
        transform_path = Path(_value_after(normalized, "--transform_path"))
        transform_payloads.append(transform_path.read_text(encoding="utf-8"))
        calls.append(normalized)
        return subprocess.CompletedProcess(normalized, 0, "stdout", "stderr")

    monkeypatch.setattr(backend, "_run", run)

    result = backend.apply_sim3(
        model_path=tmp_path / "model",
        output_path=tmp_path / "transformed",
        sim3={
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "translation": (1.0, 2.0, 3.0),
            "scale": 2.0,
        },
    )

    args = calls[0]
    assert required == ["georegister.sim3"]
    assert args[:2] == ["colmap", "model_transformer"]
    assert not Path(_value_after(args, "--transform_path")).exists()
    assert transform_payloads == ["2 0 0 1\n0 2 0 2\n0 0 2 3\n0 0 0 1\n"]
    assert result["num_points3D"] == 2


@pytest.mark.parametrize("command", COLMAP_COMMANDS)
def test_generic_colmap_bridge_accepts_every_non_gui_cli_command(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)

    result = backend.run_colmap_command(command)

    assert required == [f"colmap.{command}"]
    assert calls == [["colmap", command]]
    assert result["command"] == command
    assert result["stdout"] == "stdout"


def test_generic_colmap_bridge_normalizes_names_and_forwards_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(
        backend,
        "colmap_command_schema",
        lambda command: {
            "command": command,
            "available": True,
            "options": [
                {
                    "name": "flag",
                    "flags": ["--flag"],
                    "takes_value": True,
                    "type": "boolean",
                    "schema": {"type": "boolean"},
                },
                {
                    "name": "path",
                    "flags": ["--path"],
                    "takes_value": True,
                    "type": "string",
                    "schema": {"type": "string", "format": "path"},
                },
            ],
        },
    )

    result = backend.run_colmap_command(
        "model-analyzer",
        positional=[tmp_path / "sparse"],
        options={"flag": True, "path": tmp_path / "model", "skip": None},
    )

    args = calls[0]
    assert required == ["colmap.model_analyzer"]
    assert args[:3] == ["colmap", "model_analyzer", str(tmp_path / "sparse")]
    assert _value_after(args, "--flag") == "1"
    assert _value_after(args, "--path") == str(tmp_path / "model")
    assert "--skip" not in args
    assert result["returncode"] == 0


def test_generic_colmap_bridge_rejects_unknown_command():
    with pytest.raises(ValidationError):
        ColmapCliBackend().run_colmap_command("gui")


def test_read_reconstruction_converts_non_text_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    backend = ColmapCliBackend()

    def convert_model(input_path: Path, output_path: Path, output_type: str) -> None:
        assert input_path == tmp_path / "binary-model"
        assert output_type == "TXT"
        _write_text_model(output_path, image_name="converted.jpg")

    monkeypatch.setattr(backend, "_convert_model", convert_model)

    rec = backend.read_reconstruction(tmp_path / "binary-model")

    assert rec.num_reg_images() == 1
    assert rec.images[1].name == "converted.jpg"


def test_runtime_versions_report_missing_colmap(monkeypatch: pytest.MonkeyPatch):
    backend = ColmapCliBackend()
    monkeypatch.setattr(backend, "_find_colmap", lambda: None)
    monkeypatch.setattr(backend, "_colmap_source_sha", lambda: "source-sha")

    assert backend.runtime_versions() == {
        "backend": "0.0.1",
        "colmap_executable": "missing",
        "colmap_source_sha": "source-sha",
    }


def test_runtime_versions_include_executable_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_colmap = tmp_path / "colmap"
    fake_colmap.write_text("fake", encoding="utf-8")
    backend = ColmapCliBackend()
    monkeypatch.setattr(backend, "_find_colmap", lambda: fake_colmap)
    monkeypatch.setattr(backend, "_colmap_source_sha", lambda: "source-sha")
    monkeypatch.setattr(backend, "_colmap_help_header", lambda exe: "COLMAP help")

    versions = backend.runtime_versions()

    assert versions["colmap_executable"] == str(fake_colmap)
    assert versions["colmap_source_sha"] == "source-sha"
    assert versions["colmap_help_header"] == "COLMAP help"
    assert versions["colmap_executable_size"] == "4"
    assert "colmap_executable_mtime_ns" in versions


@pytest.mark.parametrize(
    "method_call",
    [
        lambda backend: backend.pose_graph_optimize(
            model_path=Path("model"),
            output_path=Path("out"),
            spec={},
        ),
        lambda backend: backend.convert_spherical_to_cubemap(
            input_model_path=Path("model"),
            input_image_path=Path("images"),
            output_path=Path("out"),
        ),
        lambda backend: backend.render_spherical_cubemap_images(
            input_image_path=Path("images"),
            output_path=Path("out"),
        ),
        lambda backend: backend.build_vlad_index(image_paths_by_id={}, spec={}),
        lambda backend: backend.localize_from_memory(
            sparse_dir=Path("sparse"),
            query_image=Path("query.jpg"),
            spec={},
        ),
    ],
)
def test_unsupported_protocol_features_raise_capability_errors(method_call):
    with pytest.raises(CapabilityUnavailableError):
        method_call(ColmapCliBackend())
