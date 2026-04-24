"""Bubble-to-line preprocessing used before CRNN recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class BubbleLineConfig:
    target_h: int = 32
    spacer_px: int = 8
    min_component_area: int = 10
    gap_percentile: float = 18.0
    gap_scale: float = 1.0
    min_gap_width: int = 3
    min_col_width: int = 4
    sort_right_to_left: bool = True
    horizontal_ratio: float = 1.15
    furigana_ratio: float = 0.7
    min_row_height: int = 6


def preprocess_binarize(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        15,
    )


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 255
    return out


def find_gap_runs(gap_mask: np.ndarray, min_width: int) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, is_gap in enumerate(gap_mask):
        if is_gap and start is None:
            start = i
        elif not is_gap and start is not None:
            if i - start >= min_width:
                runs.append((start, i))
            start = None
    if start is not None and len(gap_mask) - start >= min_width:
        runs.append((start, len(gap_mask)))
    return runs


def _projection_intervals(
    profile: np.ndarray, axis_length: int, cfg: BubbleLineConfig, min_span: int
) -> List[Tuple[int, int]]:
    active = profile[profile > 0]
    if active.size == 0:
        return [(0, axis_length)]

    th = float(np.percentile(active, cfg.gap_percentile) * cfg.gap_scale)
    gap_mask = profile <= th
    gaps = find_gap_runs(gap_mask, cfg.min_gap_width)

    cuts = [0]
    for g0, g1 in gaps:
        cuts.append((g0 + g1) // 2)
    cuts.append(axis_length)
    cuts = sorted(set(cuts))

    intervals: List[Tuple[int, int]] = []
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        if b - a >= min_span:
            intervals.append((a, b))

    return intervals or [(0, axis_length)]


def vertical_projection_intervals(mask: np.ndarray, cfg: BubbleLineConfig) -> List[Tuple[int, int]]:
    profile = mask.sum(axis=0).astype(np.float32)
    return _projection_intervals(profile, mask.shape[1], cfg, cfg.min_col_width)


def horizontal_projection_rows(mask: np.ndarray, cfg: BubbleLineConfig) -> List[Tuple[int, int]]:
    profile = mask.sum(axis=1).astype(np.float32)
    return _projection_intervals(profile, mask.shape[0], cfg, cfg.min_row_height)


def filter_thin_intervals(
    intervals: List[Tuple[int, int]], cfg: BubbleLineConfig
) -> List[Tuple[int, int]]:
    """Drop thin intervals, usually furigana rows/columns beside the main text."""
    if len(intervals) <= 1:
        return intervals
    spans = [b - a for a, b in intervals]
    max_span = max(spans)
    if max_span <= 0:
        return intervals
    th = max_span * cfg.furigana_ratio
    kept = [iv for iv, span in zip(intervals, spans) if span >= th]
    return kept or intervals


def crop_fg_y(mask: np.ndarray, x0: int, x1: int) -> Optional[Tuple[int, int]]:
    col = mask[:, x0:x1]
    ys = np.where(col.sum(axis=1) > 0)[0]
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1


def crop_fg_x(mask: np.ndarray, y0: int, y1: int) -> Optional[Tuple[int, int]]:
    row = mask[y0:y1, :]
    xs = np.where(row.sum(axis=0) > 0)[0]
    if xs.size == 0:
        return None
    return int(xs.min()), int(xs.max()) + 1


def rotate_and_resize_column(col_img: np.ndarray, target_h: int) -> np.ndarray:
    rot = cv2.rotate(col_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = rot.shape[:2]
    if h <= 0 or w <= 0:
        return rot
    new_w = max(1, int(round(w * (target_h / h))))
    return cv2.resize(rot, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def resize_row_to_target(row_img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = row_img.shape[:2]
    if h <= 0 or w <= 0:
        return row_img
    new_w = max(1, int(round(w * (target_h / h))))
    return cv2.resize(row_img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def _resize_whole_to_line(bgr: np.ndarray, cfg: BubbleLineConfig) -> np.ndarray:
    h, w = bgr.shape[:2]
    new_w = max(1, int(round(w * (cfg.target_h / max(1, h)))))
    return cv2.resize(bgr, (new_w, cfg.target_h), interpolation=cv2.INTER_LINEAR)


def _concat_with_spacers(pieces: List[np.ndarray], cfg: BubbleLineConfig) -> np.ndarray:
    spacer = np.full((cfg.target_h, cfg.spacer_px, 3), 255, dtype=np.uint8)
    parts: List[np.ndarray] = []
    for i, piece in enumerate(pieces):
        parts.append(piece)
        if i < len(pieces) - 1:
            parts.append(spacer)
    return np.concatenate(parts, axis=1)


def _horizontal_line_from_rows(
    bgr: np.ndarray, cleaned: np.ndarray, cfg: BubbleLineConfig
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    rows = horizontal_projection_rows(cleaned, cfg)
    rows = filter_thin_intervals(rows, cfg)
    rows = sorted(rows, key=lambda t: t[0])

    pieces: List[np.ndarray] = []
    for y0, y1 in rows:
        x_rng = crop_fg_x(cleaned, y0, y1)
        if x_rng is None:
            continue
        x0, x1 = x_rng
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        pieces.append(resize_row_to_target(crop, cfg.target_h))

    if not pieces:
        return _resize_whole_to_line(bgr, cfg), rows
    return _concat_with_spacers(pieces, cfg), rows


def bubble_to_line_image(bgr: np.ndarray, cfg: BubbleLineConfig) -> Dict[str, np.ndarray]:
    h, w = bgr.shape[:2]
    is_horizontal = w >= h * cfg.horizontal_ratio

    bw = preprocess_binarize(bgr)
    cleaned = remove_small_components(bw, cfg.min_component_area)

    if is_horizontal:
        line, rows = _horizontal_line_from_rows(bgr, cleaned, cfg)
        debug = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        for y0, y1 in rows:
            cv2.rectangle(debug, (0, y0), (debug.shape[1] - 1, y1), (0, 255, 255), 1)
        return {
            "binary": bw,
            "cleaned": cleaned,
            "debug_projection": debug,
            "line": line,
        }

    intervals = vertical_projection_intervals(cleaned, cfg)
    intervals = filter_thin_intervals(intervals, cfg)
    if cfg.sort_right_to_left:
        intervals = sorted(intervals, key=lambda x: x[0], reverse=True)
    else:
        intervals = sorted(intervals, key=lambda x: x[0])

    columns: List[np.ndarray] = []
    for x0, x1 in intervals:
        y_rng = crop_fg_y(cleaned, x0, x1)
        if y_rng is None:
            continue
        y0, y1 = y_rng
        col = bgr[y0:y1, x0:x1]
        if col.size == 0:
            continue
        columns.append(rotate_and_resize_column(col, cfg.target_h))

    if not columns:
        line = _resize_whole_to_line(bgr, cfg)
    else:
        line = _concat_with_spacers(columns, cfg)

    debug = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    for x0, x1 in intervals:
        cv2.rectangle(debug, (x0, 0), (x1, debug.shape[0] - 1), (0, 255, 255), 1)

    return {
        "binary": bw,
        "cleaned": cleaned,
        "debug_projection": debug,
        "line": line,
    }


def preprocess_file(src_path: Path, dst_path: Path, cfg: BubbleLineConfig) -> Dict[str, object]:
    bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {src_path}")

    result = bubble_to_line_image(bgr, cfg)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst_path), result["line"]):
        raise ValueError(f"Cannot write line image: {dst_path}")

    line = result["line"]
    return {
        "source": str(src_path),
        "line_image": str(dst_path),
        "source_shape": list(bgr.shape),
        "line_shape": list(line.shape),
    }

