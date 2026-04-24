"""Small helpers shared by the inference pipeline."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base_dir or repo_root()) / p


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def safe_stem(path: Path) -> str:
    keep = []
    for ch in path.stem:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "image"


def write_json(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_missing(src: Path, dst: Path) -> Dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(f"Missing source artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        same_size = src.stat().st_size == dst.stat().st_size
        same_hash = same_size and file_sha256(src) == file_sha256(dst)
        return {
            "source": str(src),
            "destination": str(dst),
            "copied": False,
            "exists": True,
            "same": same_hash,
        }
    shutil.copy2(src, dst)
    return {
        "source": str(src),
        "destination": str(dst),
        "copied": True,
        "exists": False,
        "same": True,
    }


def copy_many_if_missing(pairs: Iterable[tuple[Path, Path]]) -> List[Dict[str, Any]]:
    return [copy_if_missing(src, dst) for src, dst in pairs]

