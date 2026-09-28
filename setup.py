"""setup.py: version comes from include/gerdos/version.hpp only
(pyproject.toml declares dynamic version), and the native shared library
is compiled via CMake at build/install time so `pip install gerdos`
works from source anywhere with a compiler and CMake. Set
GERDOS_SKIP_NATIVE=1 to skip the native build (the driver then falls
back to GERDOS_LIB_DIR)."""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).parent


def read_version():
    header = ROOT / "include" / "gerdos" / "version.hpp"
    parts = {}
    for line in header.read_text().splitlines():
        match = re.match(
            r"#define GERDOS_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)",
            line.strip(),
        )
        if match:
            parts[match.group(1)] = match.group(2)
    return f"{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCH']}"


def build_native_lib(destination):
    """Configure and build the gerdos_c target, copy the .so to destination."""
    if shutil.which("cmake") is None:
        raise RuntimeError(
            "cmake is required to build the gerdos native library "
            "(apt install cmake g++); or set GERDOS_SKIP_NATIVE=1"
        )
    build_dir = Path(tempfile.mkdtemp(prefix="gerdos-native-"))
    env = dict(os.environ)
    configure = [
        "cmake", "-S", str(ROOT), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    subprocess.run(configure, env=env, check=True)
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "gerdos_c", "-j"],
        env=env,
        check=True,
    )
    built = list(build_dir.glob("**/libgerdos_c.so"))
    if not built:
        raise RuntimeError("cmake built no libgerdos_c.so")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built[0], destination / "libgerdos_c.so")
    shutil.rmtree(build_dir, ignore_errors=True)


class build_py(_build_py):
    def run(self):
        super().run()
        if os.environ.get("GERDOS_SKIP_NATIVE") == "1":
            return
        native_dir = Path(self.build_lib) / "gerdos" / "native"
        build_native_lib(native_dir)


setup(version=read_version(), cmdclass={"build_py": build_py})
