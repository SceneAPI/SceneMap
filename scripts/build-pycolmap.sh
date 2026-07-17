#!/usr/bin/env bash
set -euo pipefail

config="${1:-Release}"
build_dir="${2:-third_party/colmap/build}"
install_prefix="${SFMAPI_COLMAP_INSTALL_PREFIX:-third_party/colmap/install}"
cuda_enabled="${SFMAPI_COLMAP_CUDA:-OFF}"
cuda_archs="${SFMAPI_COLMAP_CUDA_ARCHITECTURES:-native}"
cuda_root="${SFMAPI_COLMAP_CUDATOOLKIT_ROOT:-}"
cmake_prefix_path="${SFMAPI_COLMAP_CMAKE_PREFIX_PATH:-}"
ceres_dir="${SFMAPI_COLMAP_CERES_DIR:-}"
gui_enabled="${SFMAPI_COLMAP_GUI:-ON}"
onnx_enabled="${SFMAPI_COLMAP_ONNX:-ON}"
caspar_enabled="${SFMAPI_COLMAP_CASPAR:-OFF}"
build_pycolmap="${SFMAPI_COLMAP_BUILD_PYCOLMAP:-0}"
python_executable="${PYTHON:-python}"

git submodule update --init --recursive

cmake_args=(
  -S third_party/colmap
  -B "$build_dir"
  -DCMAKE_BUILD_TYPE="$config"
  -DCMAKE_INSTALL_PREFIX="$install_prefix"
  -DCUDA_ENABLED="$cuda_enabled"
  -DGUI_ENABLED="$gui_enabled"
  -DONNX_ENABLED="$onnx_enabled"
  -DCASPAR_ENABLED="$caspar_enabled"
)

if [[ "$cuda_enabled" == "ON" ]]; then
  cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$cuda_archs")
  if [[ -n "$cuda_root" ]]; then
    cmake_args+=(-DCUDAToolkit_ROOT="$cuda_root")
  fi
fi
if [[ -n "$cmake_prefix_path" ]]; then
  cmake_args+=(-DCMAKE_PREFIX_PATH="$cmake_prefix_path")
fi
if [[ -n "$ceres_dir" ]]; then
  cmake_args+=(-DCeres_DIR="$ceres_dir")
fi

cmake "${cmake_args[@]}"
cmake --build "$build_dir" --config "$config" --parallel
cmake --install "$build_dir" --config "$config"

if [[ "$build_pycolmap" == "1" || "$build_pycolmap" == "ON" ]]; then
  old_prefix_path="${CMAKE_PREFIX_PATH:-}"
  export CMAKE_PREFIX_PATH="$(realpath "$install_prefix"):${old_prefix_path}"
  "$python_executable" -m pip install third_party/colmap --force-reinstall
  export CMAKE_PREFIX_PATH="$old_prefix_path"
fi

echo "COLMAP build finished under $build_dir"
echo "COLMAP install prefix: $install_prefix"
