"""Prepare additional naturalistic child-speech datasets.

This helper tries to download publicly available archives where possible.
For datasets that require license/authentication, see docs/dataset_instructions.md.

Usage:
  python -m src.data.prepare_extra_datasets --output_dir data/external_datasets
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from urllib.error import HTTPError, URLError


DATASET_CATALOG = {
    "myst": {
        "name": "My Science Tutor (MyST)",
        "download_urls": [
            # Public mirrors change frequently; this script attempts best-effort access.
            "https://huggingface.co/datasets/speech31/MyST/resolve/main/README.md",
        ],
        "requires_manual": True,
        "expected_subdir": "myst",
    },
    "kidstalc": {
        "name": "KidsTALC",
        "download_urls": [
            "https://huggingface.co/datasets/speech31/KidsTALC/resolve/main/README.md",
        ],
        "requires_manual": True,
        "expected_subdir": "kidstalc",
    },
    "emoreact": {
        "name": "EmoReact",
        "download_urls": [
            "https://huggingface.co/datasets/speech31/EmoReact/resolve/main/README.md",
        ],
        "requires_manual": True,
        "expected_subdir": "emoreact",
    },
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def try_download(url: str, out_path: Path) -> tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = response.read()
        out_path.write_bytes(data)
        return True, f"downloaded ({len(data)} bytes)"
    except HTTPError as e:
        return False, f"http_error_{e.code}"
    except URLError as e:
        return False, f"url_error_{getattr(e, 'reason', 'unknown')}"
    except TimeoutError:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, f"error_{type(e).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/external_datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["myst", "kidstalc", "emoreact"],
        choices=list(DATASET_CATALOG.keys()),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"output_dir": str(output_dir), "datasets": {}}
    for ds_key in args.datasets:
        cfg = DATASET_CATALOG[ds_key]
        ds_dir = output_dir / cfg["expected_subdir"]
        ds_dir.mkdir(parents=True, exist_ok=True)

        attempts = []
        downloaded_any = False
        for idx, url in enumerate(cfg["download_urls"], start=1):
            dst = ds_dir / f"url_probe_{idx}.txt"
            ok, status = try_download(url, dst)
            record = {"url": url, "ok": ok, "status": status}
            if ok:
                record["sha256"] = sha256sum(dst)
                downloaded_any = True
            attempts.append(record)

        manifest["datasets"][ds_key] = {
            "name": cfg["name"],
            "dir": str(ds_dir),
            "attempts": attempts,
            "downloaded_any": downloaded_any,
            "requires_manual": cfg["requires_manual"],
        }

    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote: {manifest_path}")
    print("If no actual corpus files were downloaded, follow docs/dataset_instructions.md")


if __name__ == "__main__":
    main()
