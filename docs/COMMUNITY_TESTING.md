# Try GERDOS on Your Hardware (5 Minutes, Nothing to Install but pip)

GERDOS revives old and small hardware. We need numbers from machines
we don't own — yours qualifies if it's old, slow, or small.

## What we need from you

On your machine, run:

```bash
apt install -y build-essential cmake python3-pip python3-venv  # once
python3 -m venv ~/gerdos-test && source ~/gerdos-test/bin/activate
pip install --no-cache-dir gerdos
python -m gerdos selftest
python -m gerdos report
```

Linux x86_64 works out of the box (verified 2026-09-29, release
0.1.2). ARM boards and macOS compile from source automatically —
keep `build-essential`/`cmake` installed and allow a few extra
minutes for the first `pip install`.

Then paste **all** output back to us. That's it — no repo, no build
knowledge, nothing to configure.

## What it does

`selftest` runs a tiny exact-math chain (normalize → select →
gather) through the runtime and checks every value bit-exact
(`0.75` everywhere), plus refusal behavior. It prints one line per
check with a machine header (CPU, RAM, OS, Python). `report` wraps
the same run as a markdown table we paste into our study record.

## What it does NOT do

No network access during tests. No GPU required (CPU-only machines
pass fully; GPU suites skip with a printed reason). No data leaves
your machine except the output you choose to paste.
