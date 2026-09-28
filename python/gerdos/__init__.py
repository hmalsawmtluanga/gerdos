"""GERDOS Python driver: thin ctypes over the C ABI. No compute in Python —
load a .gwd artifact, run it, print the verdict with evidence counts."""

import ctypes
import os
import sys


def find_library(hint=None):
    # Static archives cannot load via CDLL: the shared object only.
    # Order: explicit hint, GERDOS_LIB_DIR env, shipped package dir,
    # then the historic repo-build fallbacks (dev use).
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if hint and os.path.isfile(hint):
        return hint
    if hint:
        candidates.append(hint)
    env_dir = os.environ.get("GERDOS_LIB_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.append(os.path.join(here, "native"))
    candidates.append(here)
    if hint:
        candidates.append(os.path.join(here, os.pardir, hint))
    candidates.append(os.path.join(here, os.pardir, "build"))
    names = ("libgerdos_c.so", "gerdos_c.dll", "libgerdos_c.dylib")
    for directory in candidates:
        for name in names:
            path = os.path.normpath(os.path.join(directory, name))
            if os.path.exists(path):
                return path
    raise FileNotFoundError(
        "no shared gerdos_c library found; set GERDOS_LIB_DIR "
        "or install the built libgerdos_c.so beside the package"
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


def _machine_header():
    import datetime
    import platform
    cpu = "unknown"
    ram_kb = 0
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    ram_kb = int(line.split()[1])
                    break
    except (OSError, ValueError):
        pass
    return {
        "date": datetime.date.today().isoformat(),
        "cpu": cpu,
        "ram_mb": ram_kb // 1024,
        "os": platform.platform(),
        "python": platform.python_version(),
    }


def _check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    suffix = f" ({detail})" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return condition


def selftest(lib_hint=None, workloads_dir=None):
    """Hardware-free verification through the installed package."""
    header = _machine_header()
    print(f"board: {header['cpu']} / {header['ram_mb']} MB / {header['os']} / py{header['python']}")
    ok = True
    try:
        runtime = Runtime(lib_hint)
    except (OSError, FileNotFoundError) as exc:
        print(f"[FAIL] load library ({exc})")
        return 1
    with runtime:
        if workloads_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            bundled = os.path.join(here, "data", "signal_chain.gwd")
            repo = os.path.normpath(os.path.join(here, os.pardir, "workloads"))
            workloads_dir = here if os.path.exists(bundled) else repo
        path = os.path.join(workloads_dir, "data", "signal_chain.gwd") if workloads_dir.endswith("gerdos") else os.path.join(workloads_dir, "signal_chain.gwd")
        try:
            with open(path, "rb") as handle:
                text = handle.read()
        except OSError:
            print("[FAIL] read signal_chain.gwd")
            return 1
        ok &= _check("load", runtime.load(text) == 0)
        ok &= _check("run-3-coherent", runtime.run() == 3)
        exact = all(
            runtime.sample(data, res, i) == 0.75
            for data, res in ((901, 9002), (913, 9013), (921, 9021))
            for i in range(6)
        )
        ok &= _check("values-0.75-exact", exact)
        ok &= _check("evidence-3", runtime.evidence() == 3)
        ok &= _check("refuse-garbage", runtime.load("bogus 1 2\n") != 0)
    print("SELFTEST: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def report(lib_hint=None, workloads_dir=None):
    """One markdown block: the reproduction-table row, pasted verbatim."""
    import io
    header = _machine_header()
    print("| Field | Value |")
    print("| --- | --- |")
    print(f"| Date | {header['date']} |")
    print(f"| CPU | {header['cpu']} |")
    print(f"| RAM | {header['ram_mb']} MB |")
    print(f"| OS | {header['os']} |")
    print(f"| Python | {header['python']} |")
    buffer = io.StringIO()
    import contextlib
    with contextlib.redirect_stdout(buffer):
        code = selftest(lib_hint, workloads_dir)
    lines = [line for line in buffer.getvalue().splitlines() if line.startswith("[")]
    for line in lines:
        print(f"| check | {line} |")
    print(f"| selftest | {'PASS' if code == 0 else 'FAIL'} |")


def cli():
    sys.exit(main(sys.argv))


def main(argv):
    if len(argv) >= 2 and argv[1] == "selftest":
        hint = argv[2] if len(argv) >= 3 else None
        workloads = argv[3] if len(argv) >= 4 else None
        return selftest(hint, workloads)
    if len(argv) >= 2 and argv[1] == "report":
        hint = argv[2] if len(argv) >= 3 else None
        workloads = argv[3] if len(argv) >= 4 else None
        report(hint, workloads)
        return 0
    if len(argv) != 3:
        print("usage: python -m gerdos <build-dir> <artifact.gwd>")
        print("   or: python -m gerdos selftest [lib-dir] [workloads-dir]")
        print("   or: python -m gerdos report [lib-dir] [workloads-dir]")
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
