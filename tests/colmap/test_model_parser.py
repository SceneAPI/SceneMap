from __future__ import annotations

import sqlite3
import struct

import pytest

from scenemap.colmap.cli.backend import ColmapCliBackend as CliColmapCliBackend
from scenemap.colmap.model import read_colmap_text_model
from scenemap.colmap.native.backend import ColmapCliBackend as NativeColmapCliBackend
from scenemap.colmap.pycolmap.backend import ColmapCliBackend as PycolmapColmapCliBackend

# The three superseded repos carried this file as near-byte-identical
# copies differing only in which package's ColmapCliBackend they
# imported; the backends remain distinct implementations, so the
# database-helper tests are parametrized across all three.
PROVIDER_BACKENDS = [
    pytest.param(NativeColmapCliBackend, id="native"),
    pytest.param(PycolmapColmapCliBackend, id="pycolmap"),
    pytest.param(CliColmapCliBackend, id="cli"),
]


def test_read_colmap_text_model(tmp_path):
    (tmp_path / "cameras.txt").write_text(
        "1 SIMPLE_PINHOLE 640 480 500 320 240\n",
        encoding="utf-8",
    )
    (tmp_path / "images.txt").write_text(
        "\n".join(
            [
                "1 1 0 0 0 0 0 0 1 image one.jpg",
                "10 20 7 30 40 -1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "points3D.txt").write_text(
        "7 1.0 2.0 3.0 255 128 0 0.5 1 0\n",
        encoding="utf-8",
    )

    rec = read_colmap_text_model(tmp_path)

    assert rec.cameras[1].model_name == "SIMPLE_PINHOLE"
    assert rec.images[1].name == "image one.jpg"
    assert rec.images[1].cam_from_world.rotation.quat == (0.0, 0.0, 0.0, 1.0)
    assert rec.images[1].points2D[0].point3D_id == 7
    assert rec.images[1].points2D[1].point3D_id is None
    assert rec.points3D[7].track.elements == [(1, 0)]
    assert rec.num_reg_images() == 1


@pytest.mark.parametrize("backend_cls", PROVIDER_BACKENDS)
def test_database_helpers_decode_colmap_rows(tmp_path, backend_cls):
    db_path = tmp_path / "database.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "create table keypoints(image_id integer primary key, rows integer, cols integer, data blob)"
        )
        conn.execute(
            "create table descriptors(image_id integer primary key, rows integer, cols integer, data blob)"
        )
        conn.execute(
            "create table matches(pair_id integer primary key, rows integer, cols integer, data blob)"
        )
        conn.execute(
            "create table two_view_geometries("
            "pair_id integer primary key, rows integer, cols integer, data blob, "
            "config integer, F blob, E blob, H blob, qvec blob, tvec blob)"
        )
        conn.execute(
            "insert into keypoints values(?, ?, ?, ?)",
            (1, 2, 4, struct.pack("<8f", 1, 2, 3, 4, 5, 6, 7, 8)),
        )
        conn.execute(
            "insert into descriptors values(?, ?, ?, ?)",
            (1, 1, 4, bytes([1, 2, 3, 4])),
        )
        pair_id = 1 * 2_147_483_647 + 2
        conn.execute(
            "insert into matches values(?, ?, ?, ?)",
            (pair_id, 2, 2, struct.pack("<4I", 0, 1, 2, 3)),
        )
        conn.execute(
            "insert into two_view_geometries values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pair_id,
                1,
                2,
                struct.pack("<2I", 4, 5),
                2,
                struct.pack("<9d", *range(9)),
                None,
                None,
                None,
                None,
            ),
        )
        conn.commit()

    backend = backend_cls(executable=tmp_path / "missing-colmap")

    keypoints, descriptor_bytes, descriptor_dim = backend.read_keypoints(
        database_path=db_path,
        image_id=1,
    )
    assert keypoints == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    assert struct.unpack("<4f", descriptor_bytes) == (1.0, 2.0, 3.0, 4.0)
    assert descriptor_dim == 4

    assert list(backend.iter_correspondences(database_path=db_path)) == [(1, 2, [(0, 1), (2, 3)])]
    pairs = list(backend.iter_two_view_geometries(database_path=db_path))
    assert pairs[0][0:2] == (1, 2)
    assert pairs[0][2].config == 2
    assert pairs[0][2].inlier_matches == [(4, 5)]
    assert pairs[0][2].F == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]


@pytest.mark.parametrize("backend_cls", PROVIDER_BACKENDS)
def test_capabilities_include_upstream_cli_surface_when_executable_exists(tmp_path, backend_cls):
    fake_colmap = tmp_path / "colmap"
    fake_colmap.write_text("", encoding="utf-8")

    caps = backend_cls(executable=fake_colmap).capabilities()

    assert "pairs.spatial" in caps
    assert "pairs.vocabtree" in caps
    assert "matchers.nn-mutual" in caps
    assert "matches.verify" in caps
    assert "map.hierarchical" in caps
    assert "ba.standard" in caps
    assert "triangulate.retri" in caps
    assert "relocalize.images" in caps
    assert "recon.merge" in caps


@pytest.mark.parametrize("backend_cls", PROVIDER_BACKENDS)
def test_sim3_transform_text(backend_cls):
    backend = backend_cls()

    text = backend._sim3_to_transform_text(
        {
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "translation": (1.0, 2.0, 3.0),
            "scale": 2.0,
        }
    )

    assert text.splitlines() == [
        "2 0 0 1",
        "0 2 0 2",
        "0 0 2 3",
        "0 0 0 1",
    ]
