"""setup.py shim: version comes from include/gerdos/version.hpp only.
pyproject.toml declares dynamic version; setuptools calls this."""
import re
from pathlib import Path
from setuptools import setup

header = Path(__file__).parent / "include" / "gerdos" / "version.hpp"
parts = {}
for line in header.read_text().splitlines():
    match = re.match(r"#define GERDOS_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)", line.strip())
    if match:
        parts[match.group(1)] = match.group(2)
version = f"{parts['MAJOR']}.{parts['MINOR']}.{parts['PATCH']}"

setup(version=version)
