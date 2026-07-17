#!/usr/bin/env bash
set -euo pipefail

python_executable="${PYTHON:-python}"
output_dir="${SFMAPI_NUITKA_OUTPUT_DIR:-dist/nuitka}"
output_name="${SFMAPI_NUITKA_OUTPUT_NAME:-sfmapi-colmap-api}"
colmap_bin_dir="${SFMAPI_COLMAP_BIN_DIR:-../colmap-install-cuda-cudss/bin}"
runtime_bin_dirs="${SFMAPI_RUNTIME_BIN_DIRS:-../vcpkg_installed_colmap_cuda/x64-windows/bin}"
include_colmap_runtime="${SFMAPI_INCLUDE_COLMAP_RUNTIME:-1}"
clean="${SFMAPI_NUITKA_CLEAN:-0}"

copy_runtime_directory() {
  local source="${1:-}"
  local destination="${2:-}"
  if [[ -z "$source" || ! -d "$source" ]]; then
    return 0
  fi
  mkdir -p "$destination"
  cp -R "$source"/. "$destination"/
}

"$python_executable" -c "import sceneapi, sceneapi_map.colmap, uvicorn, fastmcp; import nuitka"

if [[ "$clean" == "1" || "$clean" == "ON" ]]; then
  rm -rf "$output_dir"
fi

"$python_executable" -m nuitka \
  --standalone \
  --assume-yes-for-downloads \
  --output-dir="$output_dir" \
  --output-filename="$output_name" \
  --include-package=sceneapi \
  --include-package=sceneapi_map.colmap \
  --include-package=uvicorn \
  --include-package=fastmcp \
  --include-package=fastapi \
  --include-package=starlette \
  --include-package=pydantic \
  --include-package=pydantic_settings \
  --include-package=sqlalchemy \
  --include-package=aiosqlite \
  --include-package=alembic \
  --include-package=anyio \
  --include-package=sse_starlette \
  --include-package=multipart \
  --include-package=PIL \
  --include-package=numpy \
  --include-package=structlog \
  --include-package=orjson \
  --include-package=prometheus_client \
  --include-package=httpx \
  --include-module=uvicorn.loops.auto \
  --include-module=uvicorn.protocols.http.auto \
  --include-module=uvicorn.protocols.websockets.auto \
  --include-module=uvicorn.lifespan.on \
  --noinclude-pytest-mode=nofollow \
  src/sceneapi_map/colmap/api_launcher.py

dist_dir="$output_dir/$output_name.dist"
actual_dist_dir="$output_dir/api_launcher.dist"
if [[ -d "$actual_dist_dir" && "$actual_dist_dir" != "$dist_dir" ]]; then
  rm -rf "$dist_dir"
  mv "$actual_dist_dir" "$dist_dir"
fi
if [[ ! -d "$dist_dir" ]]; then
  echo "Nuitka did not produce expected dist directory: $dist_dir" >&2
  exit 1
fi

if [[ "$include_colmap_runtime" == "1" || "$include_colmap_runtime" == "ON" ]]; then
  bin_dir="$dist_dir/bin"
  copy_runtime_directory "$colmap_bin_dir" "$bin_dir"
  IFS=';' read -r -a runtime_dirs <<< "$runtime_bin_dirs"
  for runtime_dir in "${runtime_dirs[@]}"; do
    copy_runtime_directory "$runtime_dir" "$bin_dir"
  done
fi

cat > "$dist_dir/sfmapi-colmap-api.env.example" <<'EOF'
SFMAPI_BACKEND=colmap_cpp_native
SFMAPI_EPHEMERAL=true
SFMAPI_DB_URL=sqlite+aiosqlite:///file::memory:?cache=shared&uri=true
SFMAPI_BLOB_BACKEND=memory
SFMAPI_QUEUE_BACKEND=inline
SFMAPI_INLINE_TASKS=true
# Set to local to serve the MCP endpoint from this API process.
SFMAPI_MCP_MODE=off
SFMAPI_MCP_MOUNT_PATH=/mcp
EOF

echo "Nuitka standalone bundle: $dist_dir"
echo "Run: $dist_dir/$output_name --host 127.0.0.1 --port 8000"
echo "MCP: $dist_dir/$output_name --mcp local --host 127.0.0.1 --port 8000"
