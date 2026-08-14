"""
Uploads local data/ contents to Google Drive via rclone, mirroring the same
relative folder structure the Colab notebooks already expect
(persona-clone/data/... -- see DRIVE_AUDIO_DIR etc. in
notebooks/colab_diarization.ipynb). Meant to keep Drive as a shared backup
between machines (T1000, Dell laptop, Colab), since data/ is entirely
gitignored and never travels via git.

**One-time setup required before this script will work** (not something
this script can do for you -- it needs your own interactive Google login):

    1. Install rclone if it isn't already (this session did it via
       `winget install --id Rclone.Rclone -e --scope user`).
    2. Run `rclone config` in a real interactive terminal (not through an
       automated tool) and create a remote:
         - name: gdrive   (or pass --remote to use a different name)
         - type: drive (Google Drive)
         - follow the browser OAuth prompt to log in and grant access
       Full guide: https://rclone.org/drive/
    3. Verify with `rclone listremotes` -- should print `gdrive:`.

**Safety note**: this script defaults to `rclone copy` (upload/update only,
never deletes anything on Drive), NOT `rclone sync` (which mirrors and
deletes remote files missing locally). Different machines hold different
subsets of data/ at any given time (e.g. the T1000 may be missing
data/raw/metadata/), so a destructive mirror from any single machine could
wipe out data another machine already backed up. Pass --mirror only if you
specifically want that machine's local state to become the source of truth
for a given folder on Drive, and only after checking that local folder
really is the complete, current copy.

Usage:
    python scripts/sync_to_drive.py                  # copy everything that exists locally
    python scripts/sync_to_drive.py --only transcripts diarized
    python scripts/sync_to_drive.py --dry-run
    python scripts/sync_to_drive.py --remote mygdrive --drive-base persona-clone-backup
"""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# name -> path relative to data/. Directories sync recursively; single files
# sync as themselves. Kept in sync with the Colab notebooks' DRIVE_*_DIR
# conventions (persona-clone/data/<same relative path>).
SYNC_TARGETS = {
    "raw_audio": DATA_DIR / "raw" / "audio",
    "raw_metadata": DATA_DIR / "raw" / "metadata",
    "raw_videos": DATA_DIR / "raw" / "videos",
    "transcripts": DATA_DIR / "transcripts",
    "diarized": DATA_DIR / "diarized",
    "exports": [
        DATA_DIR / "sadhguru_topics_and_text.txt",
        DATA_DIR / "video_title_cache.json",
    ],
}


def check_rclone(rclone_path: str, remote: str) -> None:
    if shutil.which(rclone_path) is None and not Path(rclone_path).exists():
        logger.error(
            "rclone not found at '%s'. Install it (e.g. `winget install --id "
            "Rclone.Rclone -e --scope user`) or pass --rclone-path.", rclone_path,
        )
        sys.exit(1)

    result = subprocess.run(
        [rclone_path, "listremotes"], capture_output=True, text=True,
    )
    remotes = result.stdout.split()
    if f"{remote}:" not in remotes:
        logger.error(
            "rclone remote '%s:' is not configured (found: %s). Run "
            "`rclone config` in an interactive terminal first -- see the "
            "setup instructions in this script's docstring.",
            remote, remotes or "none",
        )
        sys.exit(1)


def sync_path(rclone_path: str, remote: str, drive_base: str,
              local_path: Path, relative_dest: str,
              mirror: bool, dry_run: bool) -> bool:
    if not local_path.exists():
        logger.warning("Skipping (not present locally): %s", local_path)
        return False

    dest = f"{remote}:{drive_base}/{relative_dest}"
    verb = "sync" if mirror else "copy"
    cmd = [rclone_path, verb, str(local_path), dest, "--stats-one-line", "-v"]
    if dry_run:
        cmd.append("--dry-run")

    logger.info("%s %s -> %s", "Mirroring" if mirror else "Copying", local_path, dest)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("rclone %s failed for %s (exit code %d)", verb, local_path, result.returncode)
        return False
    return True


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sync_to_drive.py",
        description="Upload data/ to Google Drive via rclone, preserving folder structure.",
    )
    parser.add_argument("--remote", default="gdrive", help="rclone remote name (default: gdrive)")
    parser.add_argument("--drive-base", default="persona-clone",
                         help="Base folder in Drive, matching the Colab notebooks' layout (default: persona-clone)")
    parser.add_argument("--only", nargs="+", choices=list(SYNC_TARGETS.keys()),
                         help="Only sync these targets (default: all)")
    parser.add_argument("--mirror", action="store_true",
                         help="DANGEROUS: use `rclone sync` instead of `copy` -- deletes remote files "
                              "missing locally. Only use if this machine's local copy is the complete, "
                              "current one for the targets you're syncing.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would transfer without doing it.")
    parser.add_argument("--rclone-path", default="rclone", help="Path to the rclone binary if not on PATH.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    check_rclone(args.rclone_path, args.remote)

    if args.mirror:
        logger.warning(
            "--mirror requested: using `rclone sync`, which DELETES files on Drive that "
            "aren't present locally for the target(s) being synced. Ctrl+C now if that's "
            "not what you want."
        )

    targets = args.only or list(SYNC_TARGETS.keys())
    any_failed = False
    for name in targets:
        value = SYNC_TARGETS[name]
        paths = value if isinstance(value, list) else [value]
        for local_path in paths:
            relative_dest = local_path.relative_to(PROJECT_ROOT).as_posix()
            if local_path.is_file():
                relative_dest = str(Path(relative_dest).parent).replace("\\", "/")
            ok = sync_path(
                args.rclone_path, args.remote, args.drive_base,
                local_path, relative_dest, args.mirror, args.dry_run,
            )
            any_failed = any_failed or (not ok and local_path.exists())

    if any_failed:
        logger.error("One or more syncs failed -- see above.")
        return 1
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
