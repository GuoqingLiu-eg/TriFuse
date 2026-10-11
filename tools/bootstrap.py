#!/usr/bin/env python3
"""Restore pinned upstream source, apply patches and install local additions.

Does not build, download models, sign apps or access a device. Existing modified
overlay files are never overwritten. Upstream checkouts are Git-ignored.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def git(repo, *args, capture=False, check=True):
    return subprocess.run(["git", "-C", str(repo), *args], check=check, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def restore(spec):
    path = ROOT / spec["destination"]
    if not (path / ".git").exists():
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Refusing to initialize a nonempty directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        git(path, "init")
        git(path, "remote", "add", "origin", spec["url"])
        git(path, "fetch", "--depth", "1", "origin", spec["revision"])
        git(path, "checkout", "--detach", "FETCH_HEAD")
    actual = git(path, "rev-parse", "HEAD", capture=True).stdout.strip()
    if actual != spec["revision"]:
        raise RuntimeError(f"Unexpected revision at {path}: {actual}; no reset performed")
    return path


def patch(repo, patch_file):
    chunks = [chunk for chunk in re.split(rb"(?m)(?=^diff --git )", patch_file.read_bytes())
              if chunk.strip()]
    if not chunks or any(not chunk.startswith(b"diff --git ") for chunk in chunks):
        raise RuntimeError(f"Expected per-file Git diffs in {patch_file}")
    # A newer release may append file patches to an already applied patch set.
    # Check every file before changing anything; a genuine conflict must leave
    # all pending files untouched, rather than partly applying the upgrade.
    with tempfile.TemporaryDirectory(prefix="trifuse-patch-") as temporary:
        pending = []
        for index, chunk in enumerate(chunks):
            chunk_file = Path(temporary) / f"file-{index}.patch"
            chunk_file.write_bytes(chunk)
            if git(repo, "apply", "--reverse", "--check", str(chunk_file),
                   capture=True, check=False).returncode == 0:
                continue
            git(repo, "apply", "--check", str(chunk_file))
            pending.append(chunk)
        if not pending:
            print(f"Already applied: {patch_file.name}")
            return
        pending_file = Path(temporary) / "pending.patch"
        pending_file.write_bytes(b"".join(pending))
        git(repo, "apply", "--check", str(pending_file))
        git(repo, "apply", str(pending_file))


def overlay(source, target):
    for src in sorted(source.rglob("*")):
        if not src.is_file():
            continue
        dst = target / src.relative_to(source)
        if dst.exists():
            if dst.read_bytes() != src.read_bytes():
                raise RuntimeError(f"Refusing to overwrite different local content: {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submodules", action="store_true", help="Initialize selected Apple/runtime build dependencies")
    parser.add_argument("--plan", action="store_true", help="Print pins without network or filesystem changes")
    args = parser.parse_args()
    pins = json.loads((ROOT / "upstream-lock.json").read_text())["repositories"]
    names = ["executorch", "mlx"]
    if args.plan:
        print(json.dumps({name: pins[name] for name in names}, indent=2))
        return
    paths = {name: restore(pins[name]) for name in names}
    for name in ("executorch", "mlx"):
        patch(paths[name], ROOT / "patches" / (name + ".patch"))
    overlay(ROOT / "overlays/executorch", paths["executorch"])
    if args.submodules:
        git(paths["executorch"], "submodule", "update", "--init", "--depth", "1",
            "third-party/flatbuffers", "third-party/flatcc", "third-party/gflags",
            "third-party/json", "third-party/pocketfft", "third-party/ios-cmake",
            "third-party/pybind11", "third-party/prelude", "shim",
            "backends/xnnpack/third-party/FXdiv", "backends/xnnpack/third-party/cpuinfo",
            "backends/xnnpack/third-party/pthreadpool", "extension/llm/tokenizers")
        git(paths["executorch"] / "extension/llm/tokenizers", "submodule", "update", "--init", "--depth", "1",
            "third-party/abseil-cpp", "third-party/re2", "third-party/sentencepiece", "third-party/json")
        print("Selected Apple/runtime dependencies initialized; see README.md.")
    print("Pinned source restored. Builds, weights, datasets and device configuration are separate steps.")


if __name__ == "__main__":
    main()
