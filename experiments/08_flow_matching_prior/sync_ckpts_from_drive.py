"""Sync DP checkpoint(s) from a Google Drive folder to the local results dir.

Why this script exists
----------------------

Colab notebooks (Phase-2c, 2d, 3, ...) train ckpts to
``/content/drive/MyDrive/hybrid_traj_dp_phase*/``.  These persist in
the user's Drive but are not on the local Mac (and they're too big
for git: ~500 MB each).  Cell 24 in the colab notebooks calls
``files.download()`` which routes through the browser, but that path
is fragile (silent cancellations, ~100 MB hard limit on some
browsers, no automation).

This script uses ``gdown`` to pull the ckpts directly from a
shareable Google Drive folder URL into the canonical local results
directory, idempotently.

Usage
-----

1.  On https://drive.google.com, navigate to the phase folder
    (e.g. ``MyDrive/hybrid_traj_dp_phase2d/``).
2.  Right-click → Share → set "General access" to
    "Anyone with the link" (read-only is fine; revert when done).
3.  Click "Copy link".
4.  Run:

        python experiments/08_flow_matching_prior/sync_ckpts_from_drive.py \\
            --folder-url 'https://drive.google.com/drive/folders/<ID>?usp=sharing'

    By default this downloads every ``.pt`` in the folder.  Use
    ``--pattern`` to restrict (e.g. ``--pattern v2d``).

5.  After download, revert the folder's sharing back to "Restricted".

Output goes to
``Hybrid-Robot-Trajectory-Generation/experiments/08_flow_matching_prior/results/``
by default (per the Phase-2d notebook's cell-25 convention).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Default destination: the canonical results dir referenced in every
# Colab notebook's "next steps on local Mac" section.  Falls back to
# this repo's own results dir if the upstream HRTG path is missing
# (e.g. running on a fresh clone that only has HRTG-dp).
_HOME = Path.home()
_CANONICAL_DST = (
    _HOME
    / "Documents" / "SZH" / "Courseworks" / "ECE M237" / "Project"
    / "Hybrid-Robot-Trajectory-Generation"
    / "experiments" / "08_flow_matching_prior" / "results"
)
_FALLBACK_DST = Path(__file__).resolve().parent / "results"


def _default_output_dir() -> Path:
    return _CANONICAL_DST if _CANONICAL_DST.is_dir() else _FALLBACK_DST


def _check_torch_ckpt(path: Path) -> tuple[bool, str]:
    """Quick sanity check that ``path`` is a real torch checkpoint.

    Reads the first 8 bytes and looks for the PyTorch/zip magic
    (``PK\\x03\\x04`` for ZIP-format ckpts, which is what
    ``torch.save`` writes since PyTorch 1.6).  Returns
    ``(ok, message)``.  HTML error pages (which gdown silently
    writes when auth fails) start with ``<!DOC`` and are rejected.
    """
    if not path.exists():
        return False, f"missing: {path}"
    if path.stat().st_size < 1024:
        return False, f"too small ({path.stat().st_size} B): {path}"
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"PK\x03\x04"):
        return True, f"OK torch-zip ({path.stat().st_size / 1e6:.1f} MB)"
    if head.startswith(b"<!DOC") or head.startswith(b"<html"):
        return False, "looks like an HTML page (auth/share failure)"
    # Some older ckpts use the raw-pickle format -- still valid.
    if head[:2] == b"\x80\x04" or head[:2] == b"\x80\x02":
        return True, f"OK torch-pickle ({path.stat().st_size / 1e6:.1f} MB)"
    return False, f"unknown header {head!r} (size={path.stat().st_size})"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--folder-url", required=True,
        help="Google Drive folder URL (must be shared 'Anyone with the "
             "link').  Example: "
             "https://drive.google.com/drive/folders/1abc...?usp=sharing",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"Where to put the downloaded files.  Default: "
             f"{_CANONICAL_DST} (with fallback to this repo's results "
             f"dir if the canonical path is missing).",
    )
    p.add_argument(
        "--pattern", default=".pt",
        help="Only copy files whose name contains this substring "
             "(case-sensitive).  Default '.pt' = all checkpoints.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files in --output-dir even if size "
             "matches.  Default: skip if the destination already has "
             "the same byte size.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="List what would be downloaded but don't fetch or copy.",
    )
    p.add_argument(
        "--gdown", default="gdown",
        help="Path to the gdown executable.  Default: from PATH.",
    )
    args = p.parse_args()

    output_dir = (args.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sync] output dir: {output_dir}")
    print(f"[sync] folder URL: {args.folder_url}")
    print(f"[sync] pattern:    {args.pattern!r}")

    # Stage 1: gdown the whole folder into a tempdir.  gdown 6.x's
    # ``--folder`` flag preserves the original filenames and creates a
    # subdirectory named after the folder.  We don't care about that
    # subdir; we'll move files out individually.
    if args.dry_run:
        print("[sync] --dry-run: skipping gdown")
        return 0

    with tempfile.TemporaryDirectory(prefix="sync_drive_") as tmpd:
        tmp_root = Path(tmpd)
        print(f"[sync] staging into {tmp_root}")
        cmd = [
            args.gdown,
            "--folder", args.folder_url,
            "-O", str(tmp_root),
            "--quiet",
        ]
        print(f"[sync] running: {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[sync] gdown failed (rc={rc}).  Common causes:")
            print("       - folder is not shared 'Anyone with the link'")
            print("       - folder URL is wrong / file deleted")
            print("       - rate limit (try again in a minute)")
            return rc

        # Walk the tempdir and copy matching files out.
        n_copied = 0
        n_skipped = 0
        n_rejected = 0
        for src in tmp_root.rglob("*"):
            if not src.is_file():
                continue
            if args.pattern not in src.name:
                continue
            dst = output_dir / src.name
            src_size = src.stat().st_size
            if dst.exists() and not args.force:
                if dst.stat().st_size == src_size:
                    print(f"[sync] skip (same size): {dst.name}")
                    n_skipped += 1
                    continue
                else:
                    print(
                        f"[sync] OVERWRITE (size mismatch "
                        f"{dst.stat().st_size} -> {src_size}): {dst.name}"
                    )
            ok, msg = _check_torch_ckpt(src)
            if not ok:
                print(f"[sync] REJECT {src.name}: {msg}")
                n_rejected += 1
                continue
            shutil.copy2(src, dst)
            print(f"[sync] copy {src.name}  ({msg})")
            n_copied += 1

        print()
        print(f"[sync] done.  copied={n_copied}  skipped={n_skipped}  "
              f"rejected={n_rejected}")
        print(f"[sync] all files in: {output_dir}")

    # Stage 2: ls -- show what's now in the output dir
    matching = sorted(p for p in output_dir.iterdir()
                      if p.is_file() and args.pattern in p.name)
    print()
    print(f"[sync] inventory of {output_dir} matching {args.pattern!r}:")
    for p in matching:
        print(f"       {p.name:50s}  {p.stat().st_size / 1e6:>7.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
