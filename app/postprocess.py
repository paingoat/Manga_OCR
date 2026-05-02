"""Parse and normalize PaddleOCR recognition outputs."""

from __future__ import annotations

import ast
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional

IMAGE_EXT_RE = r"(?:png|jpg|jpeg|bmp|webp|gif|tif|tiff)"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _prediction_record(image_path: str, text: str, score: Optional[float]) -> Dict[str, object]:
    path = Path(image_path)
    return {
        "image_path": image_path,
        "image_name": path.name,
        "text": normalize_text(text),
        "score": score,
    }


def _parse_tuple_text(raw_tuple: str) -> Optional[tuple[str, Optional[float]]]:
    try:
        value = ast.literal_eval(raw_tuple)
    except Exception:
        match = re.search(r"\('(?P<txt>.*)'\s*,\s*(?P<score>[0-9.eE+-]+)", raw_tuple)
        if not match:
            return None
        try:
            score = float(match.group("score"))
        except ValueError:
            score = None
        return match.group("txt"), score

    if isinstance(value, tuple) and value:
        text = str(value[0])
        score = float(value[1]) if len(value) > 1 and isinstance(value[1], (float, int)) else None
        return text, score
    if isinstance(value, list) and value:
        text = str(value[0])
        score = float(value[1]) if len(value) > 1 and isinstance(value[1], (float, int)) else None
        return text, score
    if isinstance(value, str):
        return value, None
    return str(value), None


def parse_predict_lines(lines: Iterable[str]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen: set[str] = set()
    tab_result_pattern = re.compile(
        rf"^(?P<path>[^\t]+\.{IMAGE_EXT_RE})\t(?P<text>.*?)\t(?P<score>[0-9.eE+-]+)$",
        re.IGNORECASE,
    )
    patterns = [
        re.compile(
            rf"(?:Message:\s*[\"'])?Predicts\s+of\s+(?P<path>.+?\.{IMAGE_EXT_RE}):(?P<tpl>\(.*\))[\"']?$",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?P<path>[^\t]+\.{IMAGE_EXT_RE})\t(?P<tpl>\(.*\))$",
            re.IGNORECASE,
        ),
        re.compile(
            rf"result:\s*(?P<tpl>\(.*\))\t(?P<path>[^\t]+\.{IMAGE_EXT_RE})$",
            re.IGNORECASE,
        ),
    ]

    for line in lines:
        text = line.strip()
        if not text:
            continue
        tab_match = tab_result_pattern.search(text)
        if tab_match:
            image_path = tab_match.group("path").strip()
            if image_path in seen:
                continue
            try:
                score = float(tab_match.group("score"))
            except ValueError:
                score = None
            records.append(_prediction_record(image_path, tab_match.group("text"), score))
            seen.add(image_path)
            continue

        for pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            parsed = _parse_tuple_text(match.group("tpl").strip())
            if parsed is None:
                break
            image_path = match.group("path").strip()
            if image_path in seen:
                break
            pred_text, score = parsed
            records.append(_prediction_record(image_path, pred_text, score))
            seen.add(image_path)
            break

    return records


def parse_predict_file(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return parse_predict_lines(f)


def load_predictions(raw_log_path: Path, result_path: Optional[Path] = None) -> List[Dict[str, object]]:
    if result_path is not None:
        records = parse_predict_file(result_path)
        if records:
            return records
    return parse_predict_file(raw_log_path)

