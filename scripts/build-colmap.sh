#!/usr/bin/env bash
set -euo pipefail

config="${1:-Release}"
build_dir="${2:-third_party/colmap/build}"
install_prefix="${SFMAPI_COLMAP_INSTALL_PREFIX:-third_party/colmap/install}"
cuda_enabled="${SFMAPI_COLMAP_CUDA:-OFF}"
cuda_archs="${SFMAPI_COLMAP_CUDA_ARCHITECTURES:-native}"
cuda_root="${SFMAPI_COLMAP_CUDATOOLKIT_ROOT:-}"
openmp_cuda_flags="${SFMAPI_COLMAP_OPENMP_CUDA_FLAGS:-}"
cmake_prefix_path="${SFMAPI_COLMAP_CMAKE_PREFIX_PATH:-}"
ceres_dir="${SFMAPI_COLMAP_CERES_DIR:-}"
static_ceres="${SFMAPI_COLMAP_STATIC_CERES:-0}"
cudss_dir="${SFMAPI_COLMAP_CUDSS_DIR:-}"
cgal_enabled="${SFMAPI_COLMAP_CGAL:-ON}"
cgal_dir="${SFMAPI_COLMAP_CGAL_DIR:-}"
install_cgal="${SFMAPI_COLMAP_INSTALL_CGAL:-0}"
vcpkg_root="${SFMAPI_COLMAP_VCPKG_ROOT:-../vcpkg}"
vcpkg_triplet="${SFMAPI_COLMAP_VCPKG_TRIPLET:-x64-windows}"
vcpkg_install_root="${SFMAPI_COLMAP_VCPKG_INSTALL_ROOT:-../vcpkg_installed_colmap_cuda}"
gui_enabled="${SFMAPI_COLMAP_GUI:-OFF}"
onnx_enabled="${SFMAPI_COLMAP_ONNX:-ON}"
caspar_enabled="${SFMAPI_COLMAP_CASPAR:-OFF}"
build_pycolmap="${SFMAPI_COLMAP_BUILD_PYCOLMAP:-0}"
python_executable="${PYTHON:-python}"

if [[ "$cuda_enabled" == "ON" && -z "$openmp_cuda_flags" && "$(uname -s 2>/dev/null || true)" =~ MINGW|MSYS|CYGWIN ]]; then
  openmp_cuda_flags="-Xcompiler=/openmp"
fi

resolve_cmake_prefix_path() {
  local value="${1:-}"
  if [[ -z "$value" ]]; then
    return 0
  fi
  local old_ifs="$IFS"
  local parts=()
  IFS=';'
  read -r -a parts <<< "$value"
  IFS="$old_ifs"
  local out=""
  local part=""
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    if [[ -e "$part" ]]; then
      part="$(realpath "$part")"
    fi
    if [[ -n "$out" ]]; then
      out="${out};${part}"
    else
      out="$part"
    fi
  done
  printf '%s' "$out"
}

git submodule update --init --recursive

if [[ "$install_cgal" == "1" || "$install_cgal" == "ON" ]]; then
  vcpkg_exe="$vcpkg_root/vcpkg"
  if [[ ! -x "$vcpkg_exe" && -x "$vcpkg_root/vcpkg.exe" ]]; then
    vcpkg_exe="$vcpkg_root/vcpkg.exe"
  fi
  if [[ ! -x "$vcpkg_exe" ]]; then
    echo "vcpkg executable not found under $vcpkg_root" >&2
    exit 1
  fi
  "$vcpkg_exe" install "cgal:$vcpkg_triplet" "--x-install-root=$vcpkg_install_root" --recurse
  vcpkg_prefix="$(realpath "$vcpkg_install_root/$vcpkg_triplet")"
  if [[ -n "$cmake_prefix_path" ]]; then
    cmake_prefix_path="${cmake_prefix_path};${vcpkg_prefix}"
  else
    cmake_prefix_path="$vcpkg_prefix"
  fi
fi
cmake_prefix_path="$(resolve_cmake_prefix_path "$cmake_prefix_path")"
if [[ -n "$ceres_dir" && -e "$ceres_dir" ]]; then
  ceres_dir="$(realpath "$ceres_dir")"
fi
if [[ -n "$cudss_dir" && -e "$cudss_dir" ]]; then
  cudss_dir="$(realpath "$cudss_dir")"
fi
if [[ -n "$cgal_dir" && -e "$cgal_dir" ]]; then
  cgal_dir="$(realpath "$cgal_dir")"
fi

cmake_args=(
  -S third_party/colmap
  -B "$build_dir"
  -DCMAKE_BUILD_TYPE="$config"
  -DCMAKE_INSTALL_PREFIX="$install_prefix"
  -DCUDA_ENABLED="$cuda_enabled"
  -DCGAL_ENABLED="$cgal_enabled"
  -DGUI_ENABLED="$gui_enabled"
  -DONNX_ENABLED="$onnx_enabled"
  -DCASPAR_ENABLED="$caspar_enabled"
  -DGFLAGS_USE_TARGET_NAMESPACE=ON
)

if [[ "$cuda_enabled" == "ON" ]]; then
  cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$cuda_archs")
  if [[ -n "$cuda_root" ]]; then
    cmake_args+=(-DCUDAToolkit_ROOT="$cuda_root")
  fi
  if [[ -n "$openmp_cuda_flags" ]]; then
    cmake_args+=(-DOpenMP_CUDA_FLAGS="$openmp_cuda_flags")
    cmake_args+=(-DOpenMP_CUDA_LIB_NAMES="")
  fi
fi
if [[ -n "$cmake_prefix_path" ]]; then
  cmake_args+=(-DCMAKE_PREFIX_PATH="$cmake_prefix_path")
fi
if [[ -n "$ceres_dir" ]]; then
  cmake_args+=(-DCeres_DIR="$ceres_dir")
fi
if [[ -n "$cudss_dir" ]]; then
  cmake_args+=(-Dcudss_DIR="$cudss_dir")
fi
if [[ -n "$cgal_dir" ]]; then
  cmake_args+=(-DCGAL_DIR="$cgal_dir")
fi
if [[ "$static_ceres" == "1" || "$static_ceres" == "ON" ]]; then
  cmake_args+=(-DCMAKE_CXX_FLAGS="/DCERES_STATIC_DEFINE")
  cmake_args+=(-DCMAKE_CUDA_FLAGS="-DCERES_STATIC_DEFINE")
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
