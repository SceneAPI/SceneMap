from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

IMPLEMENTED_WRAPPER_OPTIONS = {
    "feature_extractor": {
        "--database_path",
        "--image_path",
        "--image_list_path",
        "--FeatureExtraction.use_gpu",
        "--SiftExtraction.max_num_features",
        "--ImageReader.single_camera",
    },
    "exhaustive_matcher": {
        "--database_path",
        "--FeatureMatching.use_gpu",
        "--ExhaustiveMatching.block_size",
    },
    "sequential_matcher": {
        "--database_path",
        "--FeatureMatching.use_gpu",
        "--SequentialMatching.overlap",
    },
    "spatial_matcher": {
        "--database_path",
        "--FeatureMatching.use_gpu",
        "--SpatialMatching.max_num_neighbors",
    },
    "vocab_tree_matcher": {
        "--database_path",
        "--FeatureMatching.use_gpu",
        "--VocabTreeMatching.vocab_tree_path",
    },
    "transitive_matcher": {
        "--database_path",
        "--TransitiveMatching.batch_size",
    },
    "geometric_verifier": {
        "--database_path",
        "--TwoViewGeometry.max_error",
    },
    "mapper": {
        "--database_path",
        "--image_path",
        "--output_path",
        "--Mapper.min_num_matches",
    },
    "hierarchical_mapper": {
        "--database_path",
        "--image_path",
        "--output_path",
    },
    "bundle_adjuster": {
        "--input_path",
        "--output_path",
        "--BundleAdjustment.refine_focal_length",
    },
    "point_triangulator": {
        "--database_path",
        "--image_path",
        "--input_path",
        "--output_path",
    },
    "image_registrator": {
        "--database_path",
        "--input_path",
        "--output_path",
    },
    "model_converter": {
        "--input_path",
        "--output_path",
        "--output_type",
    },
    "model_merger": {
        "--input_path1",
        "--input_path2",
        "--output_path",
        "--max_reproj_error",
    },
    "model_transformer": {
        "--input_path",
        "--output_path",
        "--transform_path",
    },
}


def _native_options(colmap_executable: Path, command: str) -> set[str]:
    result = subprocess.run(
        [str(colmap_executable), command, "-h"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = result.stdout + result.stderr
    return {part for part in text.replace(",", " ").split() if part.startswith("--")}


@pytest.mark.integration
@pytest.mark.needs_colmap
@pytest.mark.parametrize("command", sorted(IMPLEMENTED_WRAPPER_OPTIONS))
def test_implemented_wrapper_options_exist_in_native_colmap_help(
    colmap_executable: Path,
    command: str,
):
    native = _native_options(colmap_executable, command)

    missing = IMPLEMENTED_WRAPPER_OPTIONS[command] - native
    assert not missing, f"{command} missing expected native options: {sorted(missing)}"
