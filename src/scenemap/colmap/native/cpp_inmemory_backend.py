from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from ..model import Reconstruction
from .backend import (
    CapabilityUnavailableError,
    ColmapCliBackend,
    TwoViewGeometryRow,
    ValidationError,
)


class CppInmemoryBackend(ColmapCliBackend):
    """sfmapi backend that keeps native C++ demo features and matches in process memory."""

    name = "colmap_cpp_inmemory"
    version = "0.0.1"
    vendor = "sfmapi-colmap native C++ demo"
    _shared_stores: ClassVar[dict[str, dict[str, Any]]] = {}

    def __init__(self) -> None:
        super().__init__(executable=None)
        self._stores = self._shared_stores

    def capabilities(self) -> set[str]:
        try:
            self._require_cpp("capabilities")
        except CapabilityUnavailableError:
            return set()
        return {
            "matches.verify",
            "pairs.exhaustive",
            "pairs.sequential",
            "pairs.explicit",
            "matchers.nn-mutual",
            # Execution-mode flag: this backend runs extract / match /
            # verify entirely in process memory with no on-disk COLMAP
            # database or sparse model. It is the one COLMAP provider
            # that legitimately advertises this advisory capability.
            "compute.in_memory",
        }

    def extract_features(
        self,
        *,
        database_path: Path,
        image_root: Path,
        image_list: list[str],
        options: dict,
        progress: Any | None = None,
    ) -> dict:
        cpp = self._require_cpp("backend.actions")
        image_root = Path(image_root)
        image_paths = [image_root / image_name for image_name in image_list]
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise ValidationError(f"missing image files for C++ in-memory backend: {missing}")

        max_num_features = self._int_option(
            options,
            ("max_num_features", "SiftExtraction.max_num_features", "CppInmemory.max_num_features"),
            256,
        )
        descriptor_dim = self._int_option(
            options,
            ("descriptor_dim", "CppInmemory.descriptor_dim"),
            32,
        )
        if max_num_features < 0:
            raise ValidationError("max_num_features must be non-negative")
        if descriptor_dim <= 0:
            raise ValidationError("descriptor_dim must be positive")

        total = len(image_paths)
        self._progress(progress, "feature_extraction", current=0, total=total)
        raw_result = cpp.extract_features(
            [str(path) for path in image_paths],
            max_num_features,
            descriptor_dim,
        )
        images = self._normalize_images(raw_result["images"], image_list)
        descriptor_dim = int(raw_result["descriptor_dim"])
        total_keypoints = sum(len(image["keypoints"]) for image in images)
        self._stores[self._store_key(database_path)] = {
            "database_path": str(database_path),
            "image_root": str(image_root),
            "descriptor_dim": descriptor_dim,
            "images": images,
            "image_ids": {image["name"]: image["image_id"] for image in images},
            "matches": [],
            "verified_pairs": [],
        }
        self._progress(progress, "feature_extraction", current=total, total=total)

        return {
            "num_images": len(images),
            "num_keypoints": total_keypoints,
            "database_path": str(database_path),
            "descriptor_dim": descriptor_dim,
            "in_memory": True,
            "engine": str(raw_result["engine"]),
        }

    def match(
        self,
        *,
        database_path: Path,
        mode: str,
        options: dict,
        progress: Any | None = None,
    ) -> dict:
        normalized_mode = mode.replace("-", "_").lower()
        if normalized_mode not in {"exhaustive", "sequential", "explicit"}:
            raise CapabilityUnavailableError(
                capability="backend.actions",
                reason="the C++ in-memory backend supports exhaustive, sequential, and explicit matching",
            )

        cpp = self._require_cpp(f"pairs.{normalized_mode}")
        store = self._require_store(database_path)
        num_images = len(store["images"])
        explicit_pairs = (
            self._explicit_pair_names(options) if normalized_mode == "explicit" else None
        )
        if normalized_mode == "explicit":
            total = len(explicit_pairs or set())
        elif normalized_mode == "sequential":
            total = max(0, num_images - 1)
        else:
            total = max(0, num_images * (num_images - 1) // 2)
        self._progress(progress, "matching", current=0, total=total)
        raw_result = cpp.match_exhaustive(
            store["images"],
            self._int_option(
                options,
                ("max_distance", "FeatureMatching.max_distance", "CppInmemory.max_distance"),
                400000,
            ),
            self._bool_option(
                options,
                ("cross_check", "FeatureMatching.cross_check", "CppInmemory.cross_check"),
                True,
            ),
        )
        pairs = self._normalize_pairs(raw_result["pairs"], store)
        if normalized_mode == "sequential":
            image_positions = {image["name"]: index for index, image in enumerate(store["images"])}
            pairs = [
                pair
                for pair in pairs
                if image_positions[pair["image2"]] == image_positions[pair["image1"]] + 1
            ]
        elif normalized_mode == "explicit":
            allowed = explicit_pairs or set()
            pairs = [
                pair
                for pair in pairs
                if frozenset((str(pair["image1"]), str(pair["image2"]))) in allowed
            ]

        store["matches"] = pairs
        store["verified_pairs"] = []
        self._progress(progress, "matching", current=total, total=total)
        return {
            "database_path": str(database_path),
            "strategy": mode,
            "num_pairs": len(pairs),
            "num_matches": sum(int(pair["num_matches"]) for pair in pairs),
            "in_memory": True,
            "engine": "sfmapi_colmap._cpp_inmemory.match_exhaustive",
        }

    def verify_matches(
        self,
        *,
        database_path: Path,
        options: dict,
        progress: Any | None = None,
    ) -> dict:
        store = self._require_store(database_path)
        total = len(store["matches"])
        self._progress(progress, "geometric_verification", current=0, total=total)
        min_matches = self._int_option(
            options,
            ("min_num_inliers", "TwoViewGeometry.min_num_inliers", "CppInmemory.min_num_inliers"),
            1,
        )
        verified_pairs = [
            pair for pair in store["matches"] if int(pair["num_matches"]) >= min_matches
        ]
        store["verified_pairs"] = verified_pairs
        self._progress(progress, "geometric_verification", current=total, total=total)
        return {
            "database_path": str(database_path),
            "num_verified_pairs": len(verified_pairs),
            "num_verified_matches": sum(int(pair["num_matches"]) for pair in verified_pairs),
            "in_memory": True,
            "engine": "sfmapi_colmap._cpp_inmemory.verify_matches",
        }

    def read_keypoints(
        self,
        *,
        database_path: Path,
        image_id: int,
    ) -> tuple[list[list[float]], bytes, int]:
        store = self._stores.get(self._store_key(database_path))
        if store is None:
            return [], b"", 0

        image = self._image_by_id(store, image_id)
        if image is None:
            return [], b"", int(store["descriptor_dim"])
        descriptor_bytes = bytes(
            int(value) & 0xFF for descriptor in image["descriptors"] for value in descriptor
        )
        return image["keypoints"], descriptor_bytes, int(store["descriptor_dim"])

    def iter_correspondences(self, *, database_path: Path) -> Iterator[tuple[int, int, Any]]:
        store = self._stores.get(self._store_key(database_path))
        if store is None:
            return
        for pair in store["matches"]:
            yield (
                int(pair["image_id1"]),
                int(pair["image_id2"]),
                [(int(match[0]), int(match[1])) for match in pair["matches"]],
            )

    def iter_two_view_geometries(self, *, database_path: Path) -> Iterator[tuple[int, int, Any]]:
        store = self._stores.get(self._store_key(database_path))
        if store is None:
            return
        for pair in store["verified_pairs"]:
            yield (
                int(pair["image_id1"]),
                int(pair["image_id2"]),
                TwoViewGeometryRow(
                    config=2,
                    inlier_matches=[(int(match[0]), int(match[1])) for match in pair["matches"]],
                ),
            )

    def run_colmap_command(
        self,
        command: str,
        *,
        positional: list[str | Path] | None = None,
        options: dict | None = None,
    ) -> dict:
        raise CapabilityUnavailableError(
            capability=f"colmap.{command}",
            reason="the C++ in-memory backend does not shell out to the COLMAP CLI",
        )

    def list_colmap_commands(self) -> list[str]:
        return []

    def colmap_command_schema(self, command: str) -> dict[str, Any]:
        raise CapabilityUnavailableError(
            capability=f"colmap.{command}.schema",
            reason="the C++ in-memory backend does not expose COLMAP CLI tools",
        )

    def list_colmap_command_schemas(self) -> dict[str, dict[str, Any]]:
        return {}

    def list_backend_artifact_contracts(self) -> list[dict[str, Any]]:
        """No portable artifact contracts.

        The C++ in-memory backend keeps features, matches, and verified
        two-view geometries in process memory only -- it never
        materializes a COLMAP ``database.db`` or a sparse model
        directory, so it advertises no portable artifact I/O contracts.
        Observation sidecars are still reachable through
        ``read_keypoints`` / ``iter_correspondences`` /
        ``iter_two_view_geometries``.
        """
        return []

    def read_reconstruction(self, path: Path) -> Reconstruction:
        raise CapabilityUnavailableError(
            capability="reconstruction.read",
            reason="the C++ in-memory backend stores features and matches only",
        )

    def runtime_versions(self) -> dict[str, str]:
        versions = {
            "backend": self.version,
            "colmap": "not_used",
            "feature_store_count": str(len(self._stores)),
        }
        try:
            cpp = self._require_cpp("runtime.cpp")
        except CapabilityUnavailableError:
            versions["cpp_inmemory"] = "missing"
        else:
            versions["cpp_inmemory"] = str(cpp.version())
        return versions

    def _find_colmap(self) -> Path | None:
        return None

    def _require_colmap(self, capability: str) -> str:
        raise CapabilityUnavailableError(
            capability=capability,
            reason="the C++ in-memory backend does not use a COLMAP executable",
        )

    def _require_cpp(self, capability: str) -> Any:
        try:
            return importlib.import_module("sfmapi_colmap._cpp_inmemory")
        except (ImportError, RuntimeError) as exc:
            raise CapabilityUnavailableError(
                capability=capability,
                reason=(
                    "the sfmapi_colmap._cpp_inmemory extension is not installed; "
                    "scenemap does not build it — install a wheel built "
                    "from the superseded sfmapi_colmap repo (scikit-build-core) to "
                    "enable the C++ demo providers"
                ),
            ) from exc

    def _require_store(self, database_path: Path) -> dict[str, Any]:
        key = self._store_key(database_path)
        try:
            return self._stores[key]
        except KeyError as exc:
            raise ValidationError(
                f"no in-memory feature store for database path: {database_path}"
            ) from exc

    def _store_key(self, database_path: Path) -> str:
        return str(Path(database_path).resolve())

    def _normalize_images(self, raw_images: Any, image_list: list[str]) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        for index, raw_image in enumerate(raw_images):
            image = dict(raw_image)
            image["name"] = image_list[index]
            image["image_id"] = index + 1
            image["keypoints"] = [
                [float(value) for value in keypoint] for keypoint in image["keypoints"]
            ]
            image["descriptors"] = [
                [int(value) for value in descriptor] for descriptor in image["descriptors"]
            ]
            images.append(image)
        return images

    def _normalize_pairs(
        self,
        raw_pairs: Any,
        store: dict[str, Any],
    ) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        image_ids = store["image_ids"]
        for raw_pair in raw_pairs:
            pair = dict(raw_pair)
            pair["image_id1"] = image_ids[pair["image1"]]
            pair["image_id2"] = image_ids[pair["image2"]]
            pair["matches"] = [
                [int(match[0]), int(match[1]), int(match[2])] for match in pair["matches"]
            ]
            pair["num_matches"] = len(pair["matches"])
            pairs.append(pair)
        return pairs

    def _explicit_pair_names(self, options: dict[str, Any]) -> set[frozenset[str]]:
        pairs_spec = options.get("pairs") if isinstance(options.get("pairs"), dict) else {}
        image_pairs = options.get("image_pairs") or pairs_spec.get("image_pairs")
        if not image_pairs:
            raise CapabilityUnavailableError(
                capability="pairs.explicit",
                reason="the C++ in-memory backend requires inline image_pairs for explicit matching",
            )
        out: set[frozenset[str]] = set()
        for pair in image_pairs:
            if isinstance(pair, dict):
                image_name1 = str(pair.get("image_name1") or "")
                image_name2 = str(pair.get("image_name2") or "")
            elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
                image_name1 = str(pair[0])
                image_name2 = str(pair[1])
            else:
                raise CapabilityUnavailableError(
                    capability="pairs.explicit",
                    reason="explicit image pairs must be objects or 2-item arrays",
                )
            if not image_name1 or not image_name2 or image_name1 == image_name2:
                raise CapabilityUnavailableError(
                    capability="pairs.explicit",
                    reason="explicit image pairs require two different image names",
                )
            out.add(frozenset((image_name1, image_name2)))
        return out

    def _image_by_id(self, store: dict[str, Any], image_id: int) -> dict[str, Any] | None:
        for image in store["images"]:
            if int(image["image_id"]) == int(image_id):
                return image
        return None

    def _int_option(self, options: dict | None, names: tuple[str, ...], default: int) -> int:
        value = self._option_value(options, names, default)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"expected integer option for {names[0]}: {value!r}") from exc

    def _bool_option(self, options: dict | None, names: tuple[str, ...], default: bool) -> bool:
        value = self._option_value(options, names, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _option_value(self, options: dict | None, names: tuple[str, ...], default: Any) -> Any:
        wanted = {self._normalize_option_name(name) for name in names}
        for parts, value in self._option_items(options):
            keys = {parts[-1], ".".join(parts)}
            if len(parts) > 1:
                keys.add(f"{parts[0]}.{parts[-1]}")
            if any(self._normalize_option_name(key) in wanted for key in keys):
                return value
        return default

    def _option_items(self, options: dict | None) -> list[tuple[tuple[str, ...], Any]]:
        items: list[tuple[tuple[str, ...], Any]] = []

        def visit(prefix: tuple[str, ...], value: Any) -> None:
            if value is None:
                return
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    visit(prefix + tuple(str(nested_key).split(".")), nested_value)
            else:
                items.append((prefix, value))

        for key, value in (options or {}).items():
            visit(tuple(str(key).split(".")), value)
        return items

    def _normalize_option_name(self, value: str) -> str:
        return value.replace("_", "").replace("-", "").replace(".", "").lower()
