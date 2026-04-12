#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
WITH_PYTHON="${WITH_PYTHON:-0}"

echo "==> Installing RAIS dependencies"
if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required on macOS."
    exit 1
fi

if ! brew list catch2 >/dev/null 2>&1; then
    brew install catch2
fi

CMAKE_ARGS=(
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [[ "$WITH_PYTHON" == "1" ]]; then
    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required when WITH_PYTHON=1."
        exit 1
    fi

    if python3 -m pybind11 --cmakedir >/dev/null 2>&1; then
        PYBIND11_CMAKE_DIR="$(python3 -m pybind11 --cmakedir)"
    else
        if ! brew list pybind11 >/dev/null 2>&1; then
            brew install pybind11
        fi
        PYBIND11_CMAKE_DIR="$(brew --prefix pybind11)/share/cmake/pybind11"
    fi

    CMAKE_ARGS+=(
        -DRAIS_BUILD_PYTHON=ON
        -Dpybind11_DIR="$PYBIND11_CMAKE_DIR"
    )
fi

echo "==> Configuring CMake (${BUILD_TYPE})"
cmake "${CMAKE_ARGS[@]}"

echo "==> Building"
cmake --build "$BUILD_DIR"

echo "Done. Binaries are in $BUILD_DIR/."
