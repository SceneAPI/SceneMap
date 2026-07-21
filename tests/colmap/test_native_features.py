from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from scenemap.colmap.model import Reconstruction
from scenemap.colmap.native.backend import (
    COLMAP_COMMANDS,
    COLMAP_EXPORT_TYPES,
    CapabilityUnavailableError,
    ColmapCliBackend,
    ValidationError,
)


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, int | None]] = []

    def phase_progress(self, phase: str, *, current: int, total: int | None = None) -> None:
        self.events.append((phase, current, total))


def _capture_colmap(monkeypatch: pytest.MonkeyPatch, backend: ColmapCliBackend):
    calls: list[list[str]] = []
    required: list[str] = []

    def require_colmap(capability: str) -> str:
        required.append(capability)
        return "colmap"

    def run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        normalized = [str(arg) for arg in args]
        calls.append(normalized)
        return subprocess.CompletedProcess(normalized, 0, "stdout", "stderr")

    monkeypatch.setattr(backend, "_require_colmap", require_colmap)
    monkeypatch.setattr(backend, "_run", run)
    return calls, required


def _write_image_rows(database_path: Path, count: int) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as conn:
        conn.execute("create table images (image_id integer primary key, name text)")
        conn.executemany(
            "insert into images (image_id, name) values (?, ?)",
            [(idx, f"{idx}.jpg") for idx in range(1, count + 1)],
        )


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


def _values_after(args: list[str], option: str) -> list[str]:
    return [args[idx + 1] for idx, value in enumerate(args[:-1]) if value == option]


def test_capabilities_are_empty_when_colmap_is_missing(monkeypatch: pytest.MonkeyPatch):
    backend = ColmapCliBackend()
    monkeypatch.setattr(backend, "_find_colmap", lambda: None)

    assert backend.capabilities() == set()


def test_capabilities_cover_all_advertised_feature_paths(tmp_path: Path):
    fake_colmap = tmp_path / "colmap"
    fake_colmap.write_text("", encoding="utf-8")

    capabilities = ColmapCliBackend(executable=fake_colmap).capabilities()

    assert capabilities == {
        "features.extract.sift",
        "matches.verify",
        "pairs.exhaustive",
        "pairs.sequential",
        "pairs.spatial",
        "pairs.from_poses",
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
        "pgo.optimize",
        "recon.merge",
        "export.ply",
        "export.nvm",
        "export.colmap_text",
        "export.colmap_bin",
        "georegister.sim3",
        "georegister.gps",
        "image.undistort",
        "index.vocab_tree",
        "rigs.configure",
        "pose_priors.mapping",
    }
    assert not any(capability.startswith("colmap.") for capability in capabilities)


def test_colmap_commands_are_backend_actions_not_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fake_colmap = tmp_path / "colmap"
    fake_colmap.write_text("", encoding="utf-8")
    backend = ColmapCliBackend(executable=fake_colmap)
    monkeypatch.setattr(
        backend,
        "colmap_command_schema",
        lambda command: {
            "command": command,
            "available": True,
            "options": [
                {
                    "name": "database_path",
                    "flags": ["--database_path"],
                    "takes_value": True,
                    "type": "string",
                    "schema": {"type": "string"},
                }
            ],
            "option_count": 1,
            "schema_source": "test",
        },
    )

    actions = backend.list_backend_actions()
    action_ids = {action["action_id"] for action in actions}

    assert not any(capability.startswith("colmap.") for capability in backend.capabilities())
    assert all(f"colmap.{command}" in action_ids for command in COLMAP_COMMANDS)
    assert all(action["input_schema"] is None for action in actions)

    actions_with_schemas = backend.list_backend_actions(include_schemas=True)
    feature_action = next(
        action
        for action in actions_with_schemas
        if action["action_id"] == "colmap.feature_extractor"
    )
    assert "database_path" in feature_action["input_schema"]["properties"]

    action = backend.get_backend_action("colmap.feature_extractor")
    assert action["action_id"] == "colmap.feature_extractor"
    assert action["required_capabilities"] == []
    assert action["metadata"]["option_count"] == 1
    assert "database_path" in action["input_schema"]["properties"]


def test_backend_config_schemas_expose_stage_backend_options(
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    monkeypatch.setattr(backend, "_find_colmap", lambda: Path("colmap"))
    monkeypatch.setattr(
        backend,
        "colmap_command_schema",
        lambda command: {
            "command": command,
            "available": True,
            "schema_source": "test",
            "options": [
                {
                    "name": "database_path",
                    "schema": {"type": "string"},
                    "required": True,
                },
                {
                    "name": "SiftExtraction.peak_threshold",
                    "schema": {"type": "number"},
                },
                {
                    "name": "ImageReader.single_camera",
                    "schema": {"type": "boolean"},
                },
            ],
            "option_count": 3,
        },
    )

    compact_rows = backend.list_backend_config_schemas(include_schemas=False)
    compact_feature = next(
        row for row in compact_rows if row["config_id"] == "colmap.features.sift"
    )
    assert compact_feature["option_schema"] is None

    rows = backend.list_backend_config_schemas()
    feature = next(row for row in rows if row["config_id"] == "colmap.features.sift")

    assert feature["stage"] == "features"
    assert feature["capability"] == "features.extract.sift"
    assert feature["provider"] == "colmap"
    assert "database_path" not in feature["option_schema"]["properties"]
    assert "SiftExtraction.peak_threshold" in feature["option_schema"]["properties"]
    assert "ImageReader.single_camera" in feature["option_schema"]["properties"]


def test_colmap_backend_passes_sfmapi_backend_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("sceneapi")
    from sceneapi.backends import assert_backend_contract

    backend = ColmapCliBackend()
    monkeypatch.setattr(backend, "_find_colmap", lambda: Path("colmap"))
    monkeypatch.setattr(
        backend,
        "colmap_command_schema",
        lambda command: {
            "command": command,
            "available": True,
            "schema_source": "test",
            "options": [
                {
                    "name": "SiftExtraction.peak_threshold",
                    "schema": {"type": "number"},
                },
            ],
            "option_count": 1,
        },
    )

    assert_backend_contract(backend)


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


def test_extract_features_accepts_backend_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, _required = _capture_colmap(monkeypatch, backend)

    backend.extract_features(
        database_path=tmp_path / "db" / "database.db",
        image_root=tmp_path / "images",
        image_list=["a.jpg"],
        options={
            "backend_options": {
                "SiftExtraction.peak_threshold": 0.01,
                "ImageReader.single_camera": True,
            }
        },
    )

    args = calls[0]
    assert _value_after(args, "--SiftExtraction.peak_threshold") == "0.01"
    assert _value_after(args, "--ImageReader.single_camera") == "1"


def test_extract_features_deduplicates_colmap_option_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, _required = _capture_colmap(monkeypatch, backend)

    backend.extract_features(
        database_path=tmp_path / "db" / "database.db",
        image_root=tmp_path / "images",
        image_list=["a.jpg"],
        options={
            "use_gpu": True,
            "FeatureExtraction.use_gpu": False,
            "single_camera": False,
            "ImageReader.single_camera": True,
        },
    )

    args = calls[0]
    assert _values_after(args, "--FeatureExtraction.use_gpu") == ["0"]
    assert _values_after(args, "--ImageReader.single_camera") == ["1"]


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


def test_match_from_poses_uses_spatial_matcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)

    result = backend.match(
        database_path=tmp_path / "database.db",
        mode="from_poses",
        options={"max_num_neighbors": 12},
    )

    args = calls[0]
    # ``from_poses`` is pose-proximity pair selection, backed by COLMAP's
    # spatial_matcher; it advertises the dedicated capability id.
    assert required == ["pairs.from_poses"]
    assert args[:2] == ["colmap", "spatial_matcher"]
    assert _value_after(args, "--SpatialMatching.max_num_neighbors") == "12"
    assert result["engine"] == "colmap spatial_matcher"


def test_match_explicit_uses_matches_importer_pair_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    pairs_path = tmp_path / "pairs.txt"
    pairs_path.write_text("a.jpg b.jpg\n", encoding="utf-8")

    result = backend.match(
        database_path=tmp_path / "database.db",
        mode="explicit",
        options={"pairs": {"pairs_path": str(pairs_path)}, "matcher": {"type": "nn-mutual"}},
    )

    args = calls[0]
    assert required == ["pairs.explicit"]
    assert args[:2] == ["colmap", "matches_importer"]
    assert _value_after(args, "--match_list_path") == str(pairs_path)
    assert _value_after(args, "--match_type") == "pairs"
    assert result["engine"] == "colmap matches_importer"


def test_match_accepts_pair_and_matcher_backend_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, _required = _capture_colmap(monkeypatch, backend)

    backend.match(
        database_path=tmp_path / "database.db",
        mode="exhaustive",
        options={
            "backend_options": {
                "pairs": {"ExhaustiveMatching.block_size": 40},
                "matcher": {"SiftMatching.max_ratio": 0.75},
            }
        },
    )

    args = calls[0]
    assert _value_after(args, "--ExhaustiveMatching.block_size") == "40"
    assert _value_after(args, "--SiftMatching.max_ratio") == "0.75"


def test_match_deduplicates_colmap_option_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, _required = _capture_colmap(monkeypatch, backend)

    backend.match(
        database_path=tmp_path / "database.db",
        mode="exhaustive",
        options={
            "use_gpu": True,
            "FeatureMatching.use_gpu": False,
            "max_ratio": 0.8,
            "SiftMatching.max_ratio": 0.7,
        },
    )

    args = calls[0]
    assert _values_after(args, "--FeatureMatching.use_gpu") == ["0"]
    assert _values_after(args, "--SiftMatching.max_ratio") == ["0.7"]


def test_match_rejects_unknown_strategy():
    with pytest.raises(CapabilityUnavailableError):
        ColmapCliBackend().match(
            database_path=Path("database.db"), mode="gpu-brute-force", options={}
        )


def test_match_reports_exhaustive_progress_total(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    _capture_colmap(monkeypatch, backend)
    database_path = tmp_path / "database.db"
    _write_image_rows(database_path, 3)
    progress = RecordingProgress()

    backend.match(database_path=database_path, mode="exhaustive", options={}, progress=progress)

    assert progress.events[0] == ("matching", 0, 3)
    assert progress.events[-1] == ("matching", 3, 3)


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


def test_verify_matches_accepts_backend_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, _required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(backend, "iter_two_view_geometries", lambda *, database_path: iter([]))

    backend.verify_matches(
        database_path=tmp_path / "database.db",
        options={"backend_options": {"RANSAC.max_error": 2.0}},
    )

    assert _value_after(calls[0], "--RANSAC.max_error") == "2.0"


@pytest.mark.parametrize(
    ("kind", "command", "spec", "expected_option", "expected_value"),
    [
        (
            "incremental",
            "mapper",
            {"mapper": {"ba_refine_focal_length": True}},
            "--Mapper.ba_refine_focal_length",
            "1",
        ),
        (
            "hierarchical",
            "hierarchical_mapper",
            {"mapper": {"ba_refine_focal_length": True}},
            "--Mapper.ba_refine_focal_length",
            "1",
        ),
        (
            "global",
            "global_mapper",
            {
                "mapper": {
                    "backend": "AUTO",
                    "formulation": "AUTO",
                    "min_num_matches": 10,
                    "snapshot_frames_freq": 50,
                    "use_incremental_quality_fallback": True,
                }
            },
            "--GlobalMapper.min_num_matches",
            "10",
        ),
    ],
)
def test_run_mapping_builds_mapping_command_and_converts_models(
    kind: str,
    command: str,
    spec: dict,
    expected_option: str,
    expected_value: str,
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
        spec=spec,
    )

    args = calls[0]
    assert required == [f"map.{kind}"]
    assert args[:2] == ["colmap", command]
    assert _value_after(args, "--output_path") == str(sparse_root)
    assert _value_after(args, expected_option) == expected_value
    assert "--GlobalMapper.backend" not in args
    assert "--GlobalMapper.formulation" not in args
    assert "--GlobalMapper.snapshot_frames_freq" not in args
    assert "--GlobalMapper.use_incremental_quality_fallback" not in args
    assert conversions == [(model_dir, tmp_path / "job" / "colmap_text_models" / "0", "TXT")]
    assert summaries == [
        {"idx": 0, "num_reg_images": 1, "num_points3D": 0, "model_path": str(model_dir)}
    ]
    assert len(reconstructions) == 1


def test_run_mapping_accepts_backend_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, _required = _capture_colmap(monkeypatch, backend)

    summaries, reconstructions = backend.run_mapping(
        kind="incremental",
        db_path=tmp_path / "database.db",
        image_root=tmp_path / "images",
        sparse_root=tmp_path / "sparse",
        job_dir=tmp_path / "job",
        spec={"backend_options": {"Mapper.ba_refine_focal_length": False}},
    )

    assert _value_after(calls[0], "--Mapper.ba_refine_focal_length") == "0"
    assert summaries == []
    assert reconstructions == []


def test_run_mapping_routes_pose_priors_through_pose_prior_mapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    database_path = tmp_path / "database.db"
    _write_image_rows(database_path, 2)

    summaries, reconstructions = backend.run_mapping(
        kind="incremental",
        db_path=database_path,
        image_root=tmp_path / "images",
        sparse_root=tmp_path / "sparse",
        job_dir=tmp_path / "job",
        spec={},
        pose_priors={
            "1.jpg": {"gps": {"longitude": 7.0, "latitude": 51.0, "altitude": 120.0}},
            "2.jpg": {
                "cam_from_world": {"translation": [1.0, 2.0, 3.0]},
                "covariance": [float(i) for i in range(36)],
            },
        },
    )

    args = calls[0]
    # Priors present -> COLMAP's pose-prior-aware mapper, not plain mapper.
    assert required == ["map.incremental"]
    assert args[:2] == ["colmap", "pose_prior_mapper"]
    assert summaries == []
    assert reconstructions == []

    # The priors must have been materialized into COLMAP's pose_priors table.
    with sqlite3.connect(database_path) as conn:
        rows = conn.execute(
            "select image_id, coordinate_system from pose_priors order by image_id"
        ).fetchall()
    assert rows == [(1, 1), (2, -1)]


def test_run_mapping_without_pose_priors_uses_plain_mapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, _required = _capture_colmap(monkeypatch, backend)

    backend.run_mapping(
        kind="incremental",
        db_path=tmp_path / "database.db",
        image_root=tmp_path / "images",
        sparse_root=tmp_path / "sparse",
        job_dir=tmp_path / "job",
        spec={},
        pose_priors={},
    )

    assert calls[0][:2] == ["colmap", "mapper"]


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


def test_bundle_adjustment_accepts_backend_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, _required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(backend, "read_reconstruction", lambda path: _sample_reconstruction())

    backend.bundle_adjustment(
        model_path=tmp_path / "model",
        output_path=tmp_path / "ba",
        spec={"backend_options": {"BundleAdjustment.max_num_iterations": 20}},
    )

    assert _value_after(calls[0], "--BundleAdjustment.max_num_iterations") == "20"


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

    # Protocol signature: model_paths/output_path/sim3_aligners. The
    # optional max_reproj_error rides in via the sim3_aligners dict.
    result = backend.merge_reconstructions(
        model_paths=[tmp_path / "a", tmp_path / "b"],
        output_path=tmp_path / "merged",
        sim3_aligners={"max_reproj_error": 1.25},
    )

    args = calls[0]
    assert required == ["recon.merge"]
    assert args[:2] == ["colmap", "model_merger"]
    assert _value_after(args, "--input_path1").endswith("a")
    assert _value_after(args, "--input_path2").endswith("b")
    assert _value_after(args, "--output_path").endswith("merged")
    assert _value_after(args, "--max_reproj_error") == "1.25"
    assert result["num_sources"] == 2
    assert result["num_reg_images"] == 4
    assert result["num_points3D"] == 5


def test_merge_reconstructions_folds_three_models_pairwise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(backend, "read_reconstruction", lambda path: _sample_reconstruction(6, 7))

    result = backend.merge_reconstructions(
        model_paths=[tmp_path / "a", tmp_path / "b", tmp_path / "c"],
        output_path=tmp_path / "merged",
    )

    # COLMAP model_merger is pairwise -> two fold-left invocations.
    assert required == ["recon.merge"]
    assert len(calls) == 2
    assert calls[0][:2] == ["colmap", "model_merger"]
    assert _value_after(calls[0], "--input_path1").endswith("a")
    assert _value_after(calls[0], "--input_path2").endswith("b")
    # Second step merges the running result against the third model and
    # writes into the final output_path.
    assert _value_after(calls[1], "--input_path2").endswith("c")
    assert _value_after(calls[1], "--output_path").endswith("merged")
    assert result["num_sources"] == 3


def test_merge_reconstructions_rejects_fewer_than_two_models(tmp_path: Path):
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
                    "name": "help",
                    "flags": ["-h", "--help"],
                    "takes_value": False,
                    "type": "boolean",
                    "choices": [],
                    "schema": {"type": "boolean"},
                },
                {
                    "name": "path",
                    "flags": ["--path"],
                    "takes_value": True,
                    "type": "string",
                    "choices": [],
                    "schema": {"type": "string", "format": "path"},
                },
                {
                    "name": "log_color",
                    "flags": ["--log_color"],
                    "takes_value": True,
                    "type": "integer",
                    "choices": [],
                    "schema": {"type": "integer"},
                },
                {
                    "name": "output_type",
                    "flags": ["--output_type"],
                    "takes_value": True,
                    "type": "string",
                    "choices": ["TXT", "BIN"],
                    "schema": {"type": "string", "enum": ["TXT", "BIN"]},
                },
            ],
        },
    )

    result = backend.run_colmap_command(
        "model-analyzer",
        positional=[tmp_path / "sparse"],
        options={
            "help": True,
            "log_color": "1",
            "output_type": "TXT",
            "path": tmp_path / "model",
            "skip": None,
        },
    )

    args = calls[0]
    assert required == ["colmap.model_analyzer"]
    assert args[:3] == ["colmap", "model_analyzer", str(tmp_path / "sparse")]
    assert "--help" in args
    assert _value_after(args, "--path") == str(tmp_path / "model")
    assert _value_after(args, "--log_color") == "1"
    assert _value_after(args, "--output_type") == "TXT"
    assert "--skip" not in args
    assert result["returncode"] == 0


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"bad_option": 1}, "unknown option"),
        ({"log_color": True}, "expects integer"),
        ({"output_type": "PLY"}, "must be one of"),
    ],
)
def test_generic_colmap_bridge_validates_options_against_schema(
    options: dict,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    _calls, _required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(
        backend,
        "colmap_command_schema",
        lambda command: {
            "command": command,
            "available": True,
            "options": [
                {
                    "name": "log_color",
                    "flags": ["--log_color"],
                    "takes_value": True,
                    "type": "integer",
                    "choices": [],
                    "schema": {"type": "integer"},
                },
                {
                    "name": "output_type",
                    "flags": ["--output_type"],
                    "takes_value": True,
                    "type": "string",
                    "choices": ["TXT", "BIN"],
                    "schema": {"type": "string", "enum": ["TXT", "BIN"]},
                },
            ],
        },
    )

    with pytest.raises(ValidationError, match=message):
        backend.run_colmap_command("model_converter", options=options)


def test_generic_colmap_bridge_rejects_unknown_command():
    with pytest.raises(ValidationError):
        ColmapCliBackend().run_colmap_command("gui")


def test_colmap_command_schema_parses_defaults_choices_and_descriptions(
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    calls: list[list[str]] = []
    help_text = """COLMAP 4.1.0.dev0
  -h [ --help ]
  --project_path arg
  --log_target arg (=stderr_and_file) {stderr, stdout, file, stderr_and_file}
  --workspace_path arg                  Path to the workspace
                                        containing undistorted images
  --output_type arg                   {BIN, TXT, NVM, Bundler, VRML, PLY, R3D,
                                      CAM}
  --skip_distortion arg (=0)
"""

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, help_text, "")

    monkeypatch.setattr(backend, "_require_colmap", lambda capability: "colmap")
    monkeypatch.setattr(backend, "_run", run)

    schema = backend.colmap_command_schema("model-converter")
    options = {option["name"]: option for option in schema["options"]}

    assert schema["command"] == "model_converter"
    assert schema["available"] is True
    assert schema["schema_source"] == "colmap_help"
    assert schema["option_count"] == 6
    assert options["help"]["flags"] == ["-h", "--help"]
    assert options["help"]["schema"] == {"type": "boolean"}
    assert options["project_path"]["schema"] == {"type": "string", "format": "path"}
    assert options["log_target"]["default"] == "stderr_and_file"
    assert options["log_target"]["choices"] == ["stderr", "stdout", "file", "stderr_and_file"]
    assert options["workspace_path"]["description"] == (
        "Path to the workspace containing undistorted images"
    )
    assert options["workspace_path"]["format"] == "path"
    assert options["output_type"]["choices"][-1] == "CAM"
    assert options["skip_distortion"]["type"] == "integer"
    assert options["skip_distortion"]["default"] == 0

    assert backend.colmap_command_schema("model_converter") is schema
    assert calls == [["colmap", "model_converter", "-h"]]


def test_colmap_command_schema_rejects_unknown_command():
    with pytest.raises(ValidationError):
        ColmapCliBackend().colmap_command_schema("gui")


def test_colmap_command_schema_uses_source_fallback_for_build_disabled_command(
    monkeypatch: pytest.MonkeyPatch,
):
    backend = ColmapCliBackend()
    monkeypatch.setattr(backend, "_require_colmap", lambda capability: "colmap")

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        raise ValidationError("delaunay meshing requires CGAL")

    monkeypatch.setattr(backend, "_run", run)

    schema = backend.colmap_command_schema("delaunay_mesher")
    options = {option["name"]: option for option in schema["options"]}

    assert schema["available"] is False
    assert schema["schema_source"] == "colmap_source_fallback"
    assert schema["option_count"] == len(schema["options"])
    assert options["input_path"]["required"] is True
    assert options["input_type"]["choices"] == ["dense", "sparse"]
    assert options["DelaunayMeshing.max_proj_dist"]["default"] == 20.0
    assert options["DelaunayMeshing.num_threads"]["default"] == -1


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


@pytest.mark.parametrize(
    "method_call",
    [
        lambda backend: backend.pose_graph_optimize(
            model_path=Path("model"), output_path=Path("out"), spec={}
        ),
        lambda backend: backend.align_reconstruction(
            model_path=Path("model"), output_path=Path("out"), spec={}
        ),
        lambda backend: backend.undistort_images(
            model_path=Path("model"),
            image_root=Path("images"),
            output_path=Path("out"),
            spec={},
        ),
        lambda backend: backend.build_vocab_tree(
            database_path=Path("database.db"), output_path=Path("vocab.bin"), spec={}
        ),
        lambda backend: backend.configure_rig(
            database_path=Path("database.db"),
            spec={"rig_config_path": "rig.json"},
        ),
    ],
)
def test_native_command_wrappers_require_colmap_executable(
    method_call, monkeypatch: pytest.MonkeyPatch
):
    """The pgo / georegister-gps / undistort / vocab-tree / rig
    wrappers all shell out to the COLMAP CLI; with no executable
    reachable they must raise the capability error, not AttributeError."""
    backend = ColmapCliBackend()
    monkeypatch.setattr(backend, "_find_colmap", lambda: None)
    with pytest.raises(CapabilityUnavailableError):
        method_call(backend)


def test_pose_graph_optimize_wraps_rotation_averager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(backend, "read_reconstruction", lambda path: _sample_reconstruction(2, 5))

    result = backend.pose_graph_optimize(
        model_path=tmp_path / "model",
        output_path=tmp_path / "pgo",
        spec={"pose_graph_optimization": {"max_num_iterations": 50}},
    )

    args = calls[0]
    assert required == ["pgo.optimize"]
    assert args[:2] == ["colmap", "rotation_averager"]
    assert _value_after(args, "--input_path").endswith("model")
    assert _value_after(args, "--output_path").endswith("pgo")
    assert _value_after(args, "--max_num_iterations") == "50"
    assert result["engine"] == "colmap rotation_averager"
    assert result["num_points3D"] == 5


def test_align_reconstruction_wraps_model_aligner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    monkeypatch.setattr(backend, "read_reconstruction", lambda path: _sample_reconstruction(3, 4))

    result = backend.align_reconstruction(
        model_path=tmp_path / "model",
        output_path=tmp_path / "aligned",
        spec={"ref_images_path": str(tmp_path / "ref.txt"), "ref_is_gps": True},
    )

    args = calls[0]
    assert required == ["georegister.gps"]
    assert args[:2] == ["colmap", "model_aligner"]
    assert _value_after(args, "--input_path").endswith("model")
    assert _value_after(args, "--ref_images_path") == str(tmp_path / "ref.txt")
    assert _value_after(args, "--ref_is_gps") == "1"
    assert result["engine"] == "colmap model_aligner"


def test_undistort_images_wraps_image_undistorter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)

    result = backend.undistort_images(
        model_path=tmp_path / "model",
        image_root=tmp_path / "images",
        output_path=tmp_path / "undist",
        spec={"max_image_size": 1600, "copy_policy": "soft-link"},
    )

    args = calls[0]
    assert required == ["image.undistort"]
    assert args[:2] == ["colmap", "image_undistorter"]
    assert _value_after(args, "--image_path").endswith("images")
    assert _value_after(args, "--input_path").endswith("model")
    assert _value_after(args, "--max_image_size") == "1600"
    assert _value_after(args, "--copy_policy") == "soft-link"
    assert result["model_path"].endswith("sparse")


def test_build_vocab_tree_wraps_vocab_tree_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)

    result = backend.build_vocab_tree(
        database_path=tmp_path / "database.db",
        output_path=tmp_path / "vocab.bin",
        spec={"num_visual_words": 4096},
    )

    args = calls[0]
    assert required == ["index.vocab_tree"]
    assert args[:2] == ["colmap", "vocab_tree_builder"]
    assert _value_after(args, "--database_path").endswith("database.db")
    assert _value_after(args, "--vocab_tree_path").endswith("vocab.bin")
    assert _value_after(args, "--num_visual_words") == "4096"
    assert result["engine"] == "colmap vocab_tree_builder"


def test_configure_rig_wraps_rig_configurator_with_inline_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    backend = ColmapCliBackend()
    calls, required = _capture_colmap(monkeypatch, backend)
    rig_payloads: list[str] = []

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        normalized = [str(arg) for arg in args]
        rig_path = Path(_value_after(normalized, "--rig_config_path"))
        rig_payloads.append(rig_path.read_text(encoding="utf-8"))
        calls.append(normalized)
        return subprocess.CompletedProcess(normalized, 0, "stdout", "stderr")

    monkeypatch.setattr(backend, "_run", run)

    result = backend.configure_rig(
        database_path=tmp_path / "database.db",
        spec={"rig_config": [{"cameras": [{"image_prefix": "cam0/"}]}]},
    )

    args = calls[0]
    assert required == ["rigs.configure"]
    assert args[:2] == ["colmap", "rig_configurator"]
    assert not Path(_value_after(args, "--rig_config_path")).exists()
    assert "image_prefix" in rig_payloads[0]
    assert result["engine"] == "colmap rig_configurator"


def test_configure_rig_requires_a_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = ColmapCliBackend()
    _capture_colmap(monkeypatch, backend)
    with pytest.raises(ValidationError, match="rig_config"):
        backend.configure_rig(database_path=tmp_path / "database.db", spec={})
