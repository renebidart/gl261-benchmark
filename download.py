#!/usr/bin/env python3
"""Download the GL261 mouse brain tumor ultrasound dataset from Figshare.

Article: https://figshare.com/articles/dataset/27237894
Destination: data/raw/

Usage:
    python download.py
    python download.py --dry-run
    python download.py --skip-videos
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
FIGSHARE_API = "https://api.figshare.com/v2/articles/27237894/files"
DEST_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = DEST_DIR / "download_manifest.json"

MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds
CHUNK_SIZE = 1 << 20  # 1 MiB


# -- helpers -----------------------------------------------------------------


def fetch_file_list() -> list[dict]:
    """Query Figshare API for the article's file listing (paginated)."""
    all_files = []
    page = 1
    while True:
        resp = requests.get(
            FIGSHARE_API,
            params={"page": page, "page_size": 100},
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_files.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return all_files


def download_file(url: str, dest: Path, expected_size: int) -> bool:
    """Download a single file with progress bar and retry logic.

    Returns True if the file was downloaded, False if skipped (already exists).
    """
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"  SKIP (already exists, size matches): {dest.name}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", expected_size))
            with open(dest, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=f"  {dest.name}",
                leave=True,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))
            return True

        except (requests.RequestException, IOError) as exc:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"  RETRY {attempt}/{MAX_RETRIES} after error: {exc}. "
                      f"Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to download {dest.name} after {MAX_RETRIES} attempts"
                ) from exc
    return True  # unreachable


def extract_zip(zip_path: Path, dest_dir: Path, skip_videos: bool = False) -> int:
    """Extract a ZIP archive. Returns number of files extracted."""
    extracted = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in members:
            if member.endswith("/"):
                continue
            if skip_videos and member.lower().endswith(".mp4"):
                continue
            zf.extract(member, dest_dir)
            extracted += 1
    return extracted


def validate_dataset(dest_dir: Path) -> dict:
    """Count images and masks, check alignment."""
    image_files = sorted(dest_dir.rglob("Images/*.png"))
    mask_files = sorted(dest_dir.rglob("Masks/*.png"))

    n_images = len(image_files)
    n_masks = len(mask_files)

    def _rel_key(p: Path, strip_mask: bool = False) -> str:
        stem = p.stem
        if strip_mask and stem.endswith("_mask"):
            stem = stem[:-5]
        return str(p.relative_to(dest_dir).parent.parent / stem)

    image_keys = {_rel_key(f) for f in image_files}
    mask_keys = {_rel_key(f, strip_mask=True) for f in mask_files}

    missing_masks = image_keys - mask_keys
    missing_images = mask_keys - image_keys

    return {
        "n_images": n_images,
        "n_masks": n_masks,
        "match": n_images == n_masks and n_images > 0 and not missing_masks and not missing_images,
        "missing_masks": len(missing_masks),
        "missing_images": len(missing_images),
    }


def write_manifest(files: list[dict], dest: Path) -> None:
    """Write download manifest with URLs, sizes, and timestamp."""
    manifest = {
        "source": "https://figshare.com/articles/dataset/27237894",
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "name": f["name"],
                "size_bytes": f["size"],
                "download_url": f["download_url"],
                "md5": f.get("computed_md5") or f.get("supplied_md5", ""),
            }
            for f in files
        ],
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Manifest written to {dest}")


# -- main --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download GL261 mouse brain tumor ultrasound dataset from Figshare."
    )
    parser.add_argument("--dest", type=Path, default=DEST_DIR,
                        help="Destination directory (default: data/raw)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview files without downloading.")
    parser.add_argument("--skip-videos", action="store_true",
                        help="Skip .mp4 files during extraction to save disk space.")
    args = parser.parse_args()

    dest_dir = args.dest

    # 1. Fetch file list
    print("Querying Figshare API...")
    files = fetch_file_list()
    total_bytes = sum(f["size"] for f in files)
    print(f"Found {len(files)} files, total {total_bytes / 1e9:.2f} GB\n")

    for f in files:
        print(f"  {f['name']:55s}  {f['size'] / 1e6:>8.1f} MB")
    print()

    if args.dry_run:
        print("DRY RUN -- no files will be downloaded.")
        return

    # 2. Download
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    for f in files:
        dest_path = dest_dir / f["name"]
        was_downloaded = download_file(f["download_url"], dest_path, f["size"])
        if was_downloaded:
            downloaded += 1
        else:
            skipped += 1

    # 3. Extract ZIPs
    extracted_total = 0
    zip_files = sorted(dest_dir.glob("*.zip"))
    print(f"\nExtracting {len(zip_files)} ZIP archives...")
    for zp in zip_files:
        print(f"  Extracting {zp.name}...")
        n = extract_zip(zp, dest_dir, skip_videos=args.skip_videos)
        extracted_total += n
        print(f"    -> {n} files")

    # 4. Validate
    print("\nValidating dataset...")
    val = validate_dataset(dest_dir)
    status = "OK" if val["match"] else "MISMATCH"
    print(f"  Images/: {val['n_images']} PNGs")
    print(f"  Masks/:  {val['n_masks']} PNGs")
    if val["missing_masks"]:
        print(f"  WARNING: {val['missing_masks']} images have no matching mask")
    if val["missing_images"]:
        print(f"  WARNING: {val['missing_images']} masks have no matching image")
    print(f"  Validation: {status}")

    # 5. Write manifest
    manifest_path = dest_dir / "download_manifest.json"
    write_manifest(files, manifest_path)

    # 6. Summary
    print(f"\n{'=' * 60}")
    print(f"  Files downloaded:  {downloaded}")
    print(f"  Files skipped:     {skipped}")
    print(f"  Files extracted:   {extracted_total}")
    print(f"  Images validated:  {val['n_images']}")
    print(f"  Masks validated:   {val['n_masks']}")
    print(f"  Status:            {status}")
    print(f"  Destination:       {dest_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
