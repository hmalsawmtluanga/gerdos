"""GERDOS Python driver: thin ctypes over the C ABI. No compute in Python —
load a .gwd artifact, run it, print the verdict with evidence counts."""

import ctypes
import os
import sys


def find_library(build_dir):
    # Static archives cannot load via CDLL: prefer the shared object;
    # fall back to building one is the caller's job (see README note).
    # Also resolve build_dir against the caller's cwd AND the driver's
    # own location (repoane/python) so `python -m gerdos build ...`
    # works from the repo root.
    candidates = [build_dir]
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, os.pardir, build_dir))
    candidates.append(os.path.join(here, os.pardir, "build"))
    names = ("libgerdos_c.so", "gerdos_c.dll", "libgerdos_c.dylib")
    for directory in candidates:
        for name in names:
            path = os.path.normpath(os.path.join(directory, name))
            if os.path.exists(path):
                return path
    raise FileNotFoundError(
        "no shared gerdos_c library found (build with -DBUILD_SHARED_LIBS=ON)"
    )


class Runtime:
    def __init__(self, build_dir):
        self.lib = ctypes.CDLL(find_library(build_dir))
        lib = self.lib
        lib.gerdos_create.restype = ctypes.c_void_p
        lib.gerdos_destroy.argtypes = [ctypes.c_void_p]
        lib.gerdos_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.gerdos_load.restype = ctypes.c_int
        lib.gerdos_run.argtypes = [ctypes.c_void_p]
        lib.gerdos_run.restype = ctypes.c_int
        lib.gerdos_sample.argtypes = [
            ctypes.c_void_p, ctypes.c_ulonglong,
            ctypes.c_ulonglong, ctypes.c_size_t,
        ]
        lib.gerdos_sample.restype = ctypes.c_float
        lib.gerdos_evidence.argtypes = [ctypes.c_void_p]
        lib.gerdos_evidence.restype = ctypes.c_ulonglong
        self.handle = lib.gerdos_create()
        if not self.handle:
            raise RuntimeError("gerdos_create failed")

    def close(self):
        if self.handle:
            self.lib.gerdos_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def load(self, text):
        if isinstance(text, str):
            text = text.encode()
        return self.lib.gerdos_load(self.handle, text)

    def load_file(self, path):
        with open(path, "rb") as handle:
            return self.load(handle.read())

    def run(self):
        return self.lib.gerdos_run(self.handle)

    def sample(self, data, residency, index=0):
        return self.lib.gerdos_sample(self.handle, data, residency, index)

    def evidence(self):
        return self.lib.gerdos_evidence(self.handle)


def main(argv):
    if len(argv) != 3:
        print("usage: python -m gerdos <build-dir> <artifact.gwd>")
        return 1
    with Runtime(argv[1]) as runtime:
        refused = runtime.load_file(argv[2])
        if refused != 0:
            print(f"refused at line {refused}")
            return 1
        coherent = runtime.run()
        print(f"coherent={coherent} evidence={runtime.evidence()}")
        return 0 if coherent >= 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
