# %% [markdown]
# # `final_data.ipynb` — Chuẩn bị dữ liệu (bubble + line)
# 
# Notebook chạy trên **Kaggle**: tải Manga109-s (HF), build nguồn `rec`, xuất **hai bộ** ảnh crop (bubble) và ảnh đã preprocess (line), **đúng 60.000** mẫu sau dedup, chia **train / val / test = 80% / 10% / 10%**.
# 
# - **Seed `42`**: `random` và `numpy` dùng cùng seed để tái lập shuffle/split.
# - **Output gốc**: `/kaggle/working/final_data/` (`bubble_dataset/`, `line_dataset/`).
# - **Lọc nhãn**: bỏ mẫu có nhãn ≤ 1 ký tự, bỏ mẫu có nhãn `> 80` ký tự (khớp giới hạn CTC của SVTR với `REC_IMAGE_SHAPE=[3,32,320]` → `T=80`), và bỏ mẫu chỉ gồm dấu câu/ký hiệu (không chứa chữ/kana/kanji/chữ số).
# - **Khử trùng lặp**: khử trùng lặp `(rel_path, text)` trước khi lấy mẫu và khử trùng lặp theo nội dung ảnh (MD5) khi ghi.
# - **Loại furigana**: áp lọc furigana cho **cả ảnh dọc lẫn ảnh ngang** (drop interval có span `< furigana_ratio × max_span`).
# - **Over-sample 1.1×** rồi **cap về 60k**: xử lý `int(MAX_SAMPLES × OVERSAMPLE_FACTOR)` mẫu vào thư mục `_staging/`, cắt đúng 60k, rồi mới split 80/10/10 và rename sang `train/val/test/`.
# - **Không** cài Paddle/torch hay clone PaddleOCR — chỉ phục vụ chuẩn bị dữ liệu.
# 

# %% [markdown]
# ### Bước 1 — Cài thư viện tối thiểu
# 
# Cài `huggingface_hub` (tải dataset), OpenCV, Pillow, tqdm. Không cài Paddle/torch.
# 

# %%
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"]
)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "huggingface_hub",
        "opencv-python-headless",
        "pillow",
        "tqdm",
    ]
)
print("Dependencies OK.")


# %% [markdown]
# ### Bước 2 — Đăng nhập Hugging Face (Kaggle Secrets)
# 
# Dùng secret `HF_TOKEN` (Add-ons → Secrets) để tải dataset private / gated nếu cần.
# 

# %%
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("HF_TOKEN")
if secret_value_0 and str(secret_value_0).strip():
    login(str(secret_value_0).strip())
    print("HF login OK (HF_TOKEN).")
else:
    print("HF_TOKEN missing or empty — snapshot download may fail if repo requires auth.")


# %% [markdown]
# ### Bước 3 — Import, seed 42, đường dẫn output
# 
# Đặt `OUTPUT_ROOT`, `BUBBLE_ROOT`, `LINE_ROOT`; tạo `train/`, `val/`, `test/` cho mỗi bộ.
# 

# %%
import os
import re
import json
import random
import shutil
import hashlib
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_ROOT = Path("/kaggle/working/final_data")
BUBBLE_ROOT = OUTPUT_ROOT / "bubble_dataset"
LINE_ROOT = OUTPUT_ROOT / "line_dataset"

MAX_SAMPLES = 60_000
OVERSAMPLE_FACTOR = 1.1  # process 10% more to absorb MD5 dedup / decode loss, then cap to MAX_SAMPLES
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

for root in (BUBBLE_ROOT, LINE_ROOT):
    for sub in ("train", "val", "test"):
        (root / sub).mkdir(parents=True, exist_ok=True)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("SEED =", SEED)
print("OUTPUT_ROOT =", OUTPUT_ROOT)
print("BUBBLE_ROOT =", BUBBLE_ROOT)
print("LINE_ROOT   =", LINE_ROOT)


# %% [markdown]
# ### Bước 4 — Tải dataset Manga109-s từ Hugging Face
# 
# Nếu file zip đã có sẵn trong thư mục làm việc thì bỏ qua tải lại.
# 

# %%
from huggingface_hub import snapshot_download

WORK_DIR = Path("/kaggle/working")
HF_DOWNLOAD_DIR = WORK_DIR / "manga109s_dataset"
EXTRACT_DIR = WORK_DIR / "manga109_extracted"
DATASET_ROOT = EXTRACT_DIR / "Manga109s_released_2023_12_07"
ZIP_PATH = HF_DOWNLOAD_DIR / "Manga109s_released_2023_12_07.zip"

HF_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    if "secret_value_0" not in globals() or not str(secret_value_0).strip():
        raise RuntimeError(
            "HF token not found. Set HF_TOKEN in Kaggle Secrets or place zip manually."
        )
    snapshot_download(
        repo_id="hal-utokyo/Manga109-s",
        repo_type="dataset",
        local_dir=str(HF_DOWNLOAD_DIR),
        token=str(secret_value_0).strip(),
    )

print("HF download dir:", HF_DOWNLOAD_DIR)
print("Zip exists:", ZIP_PATH.exists(), ZIP_PATH)


# %% [markdown]
# ### Bước 5 — Giải nén zip Manga109-s
# 
# Chỉ giải nén khi thư mục đích trống hoặc chưa có nội dung.
# 

# %%
if ZIP_PATH.exists() and (not DATASET_ROOT.exists() or not any(DATASET_ROOT.iterdir())):
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
print("DATASET_ROOT:", DATASET_ROOT)
if DATASET_ROOT.exists():
    print("Sample entries:", sorted([p.name for p in DATASET_ROOT.iterdir()])[:15])
else:
    print("DATASET_ROOT not found — skip if you use pre-built rec only.")


# %% [markdown]
# ### Bước 6 — Tìm hoặc tạo thư mục `rec` (crop + `rec_gt_train.txt`)
# 
# Ưu tiên thư mục đã có `rec_gt_train.txt`; nếu không, build từ XML + ảnh trong `DATASET_ROOT`.
# 

# %%
import xml.etree.ElementTree as ET
from PIL import Image


def _find_existing_rec_root() -> Optional[Path]:
    candidate_roots = [
        Path("/kaggle/working/train_data/rec"),
        Path("/kaggle/working/manga109/rec"),
    ]
    if "DATASET_ROOT" in globals() and Path(DATASET_ROOT).exists():
        candidate_roots.extend(
            [
                Path(DATASET_ROOT) / "rec",
                Path(DATASET_ROOT) / "train_data" / "rec",
                Path(DATASET_ROOT),
            ]
        )
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        for rec_label in kaggle_working.rglob("rec_gt_train.txt"):
            candidate_roots.append(rec_label.parent)
    seen = set()
    uniq = []
    for p in candidate_roots:
        k = str(p)
        if k not in seen:
            uniq.append(p)
            seen.add(k)
    for c in uniq:
        probe = c / "rec_gt_train.txt"
        if probe.exists():
            return c
    return None


def _build_rec_from_manga109s(dataset_root: Path, out_root: Path, max_samples: int = 150_000) -> Path:
    annotations_dir = dataset_root / "annotations"
    images_dir = dataset_root / "images"
    if not annotations_dir.exists() or not images_dir.exists():
        raise FileNotFoundError(f"Missing annotations/images under {dataset_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    out_train = out_root / "train"
    out_train.mkdir(parents=True, exist_ok=True)
    out_label = out_root / "rec_gt_train.txt"
    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise RuntimeError(f"No XML in {annotations_dir}")
    num_written = 0
    with open(out_label, "w", encoding="utf-8") as fw:
        for xml_path in tqdm(xml_files, desc="Parse books"):
            book_title = xml_path.stem
            try:
                root = ET.parse(xml_path).getroot()
            except Exception:
                continue
            for page in root.findall(".//page"):
                page_idx = page.get("index")
                if page_idx is None:
                    continue
                img_path = images_dir / book_title / f"{int(page_idx):03d}.jpg"
                if not img_path.exists():
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                for text_obj in page.findall("text"):
                    try:
                        xmin = int(text_obj.get("xmin"))
                        ymin = int(text_obj.get("ymin"))
                        xmax = int(text_obj.get("xmax"))
                        ymax = int(text_obj.get("ymax"))
                        label = (text_obj.text or "").strip().replace("\n", "").replace("\t", " ")
                        if not label or (xmax - xmin) < 12 or (ymax - ymin) < 12:
                            continue
                        crop = img.crop((xmin, ymin, xmax, ymax))
                        num_written += 1
                        out_name = f"word_{num_written:07d}.png"
                        crop.save(out_train / out_name)
                        fw.write(f"train/{out_name}\t{label}\n")
                        if max_samples and num_written >= max_samples:
                            return out_root
                    except Exception:
                        continue
    return out_root


resolved = _find_existing_rec_root()
if resolved is None:
    if "DATASET_ROOT" in globals() and Path(DATASET_ROOT).exists():
        auto_out = Path("/kaggle/working/train_data/rec")
        print("Building rec from Manga109-s XML...")
        resolved = _build_rec_from_manga109s(Path(DATASET_ROOT), auto_out, max_samples=150_000)
    else:
        raise FileNotFoundError(
            "No rec_gt_train.txt and DATASET_ROOT missing. Run HF download + extract first."
        )

DATA_ROOT = Path(str(resolved))
SRC_LABEL = DATA_ROOT / "rec_gt_train.txt"
SRC_IMAGE_ROOT = DATA_ROOT
print("DATA_ROOT =", DATA_ROOT)
print("SRC_LABEL exists:", SRC_LABEL.exists())

with open(SRC_LABEL, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(line.rstrip())
        if i >= 3:
            break


# %% [markdown]
# ### Bước 7 — Preprocess bubble → line (lọc furigana cho cả hai chiều)
# 
# Chung cho hai nhánh: dùng `filter_thin_intervals` để loại các interval có span `< furigana_ratio × max_span` (mặc định `0.7`) — tương ứng với furigana vì chúng luôn mảnh/ngắn hơn phần chữ chính.
# 
# - **Ảnh ngang** (`width >= height * horizontal_ratio`, mặc định 1.15): projection theo trục ngang để tách thành các **dòng**, bỏ dòng furigana (height nhỏ), rồi ghép các dòng chính còn lại theo chiều ngang (có spacer trắng) thành một line duy nhất.
# - **Ảnh dọc**: projection theo trục dọc để tách thành các **cột**, bỏ cột furigana (width nhỏ), sau đó sort phải→trái, xoay từng cột CCW và ghép như pipeline gốc.
# 

# %%
from dataclasses import dataclass


@dataclass
class BubbleLineConfig:
    target_h: int = 32
    spacer_px: int = 8
    min_component_area: int = 10
    gap_percentile: float = 18.0
    gap_scale: float = 1.0
    min_gap_width: int = 3
    min_col_width: int = 4
    sort_right_to_left: bool = True
    # width >= height * horizontal_ratio => treat as horizontal line (row projection)
    horizontal_ratio: float = 1.15
    # Rows whose height < max_row_height * furigana_ratio are considered furigana and dropped
    furigana_ratio: float = 0.7
    min_row_height: int = 6


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def preprocess_binarize(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        15,
    )
    return bw


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
    start = None
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

    if not intervals:
        intervals = [(0, axis_length)]
    return intervals


def vertical_projection_intervals(mask: np.ndarray, cfg: BubbleLineConfig) -> List[Tuple[int, int]]:
    profile = mask.sum(axis=0).astype(np.float32)
    return _projection_intervals(profile, mask.shape[1], cfg, cfg.min_col_width)


def horizontal_projection_rows(mask: np.ndarray, cfg: BubbleLineConfig) -> List[Tuple[int, int]]:
    profile = mask.sum(axis=1).astype(np.float32)
    return _projection_intervals(profile, mask.shape[0], cfg, cfg.min_row_height)


def filter_thin_intervals(
    intervals: List[Tuple[int, int]], cfg: BubbleLineConfig
) -> List[Tuple[int, int]]:
    """Drop intervals whose span (b - a) is below ``furigana_ratio * max_span``.

    Works for both row intervals (horizontal branch) and column intervals (vertical branch).
    Rationale: furigana forms a thinner row above / column beside the main text.
    """
    if len(intervals) <= 1:
        return intervals
    spans = [b - a for a, b in intervals]
    max_span = max(spans)
    if max_span <= 0:
        return intervals
    th = max_span * cfg.furigana_ratio
    kept = [iv for iv, s in zip(intervals, spans) if s >= th]
    return kept if kept else intervals


filter_furigana_rows = filter_thin_intervals


def crop_fg_y(mask: np.ndarray, x0: int, x1: int) -> Optional[Tuple[int, int]]:
    col = mask[:, x0:x1]
    ys = np.where(col.sum(axis=1) > 0)[0]
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return y0, y1


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
    """Row-projection split on horizontal bubble, drop furigana rows, concatenate main rows."""
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


# %% [markdown]
# ### Bước 8 — Lọc nhãn, khử trùng lặp, over-sample 1.1×, cap 60k, split 80/10/10
# 
# Pipeline:
# 
# 1. **Lọc nhãn** (`filter_and_deduplicate`): bỏ mẫu có nhãn `< 2` ký tự, bỏ mẫu có nhãn `> 80` ký tự (khớp giới hạn CTC của SVTR `REC_IMAGE_SHAPE=[3,32,320]` → `T=80`), bỏ mẫu chỉ chứa dấu câu/ký hiệu (không có chữ/kana/kanji/chữ số — ví dụ `.....` hoặc `!?.`), khử cặp `(rel_path, text)` trùng.
# 2. **Shuffle** với `SEED = 42`, lấy `int(MAX_SAMPLES × OVERSAMPLE_FACTOR)` mẫu đầu tiên (mặc định 1.1× → 66 000).
# 3. **Process vào staging** (`process_items_to_staging`): với mỗi mẫu, khử trùng lặp theo **MD5 của nội dung ảnh**, chạy preprocess bubble → line, ghi cặp ảnh vào `<root>/_staging/stage_NNNNNNN.png`.
# 4. **Cap + split + rename** (`finalize_split_from_staging`): cắt staging về đúng `MAX_SAMPLES`, chia 80/10/10 giữ nguyên thứ tự đã shuffle, rename từng file sang `<root>/<split>/<split>_NNNNNN.png`, dọn thư mục `_staging/`.
# 5. Hai bộ (`bubble_dataset`, `line_dataset`) dùng **cùng thứ tự** mẫu và **cùng tên file** trong `train|val|test`.
# 

# %%
# Unicode ranges that count as "textual" characters (letters / digits / kana / kanji).
_TEXT_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x31F0, 0x31FF),  # Katakana phonetic extensions
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),  # Halfwidth Katakana
)

MIN_LABEL_CHARS = 2
MAX_LABEL_CHARS = 80  # upper cap matching SVTR CTC T=80 for REC_IMAGE_SHAPE=[3,32,320]


def has_text_char(text: str) -> bool:
    """True if ``text`` has at least one letter / digit / kana / kanji character."""
    for ch in text:
        code = ord(ch)
        if any(lo <= code <= hi for lo, hi in _TEXT_RANGES):
            return True
        if ch.isalnum():
            return True
    return False


def is_label_valid(
    text: str,
    min_chars: int = MIN_LABEL_CHARS,
    max_chars: int = MAX_LABEL_CHARS,
) -> bool:
    return min_chars <= len(text) <= max_chars and has_text_char(text)


def read_label_file(label_path: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            rel_path, text = line.split("\t", 1)
            text = normalize_text(text)
            if not text:
                continue
            items.append((rel_path, text))
    return items


def filter_and_deduplicate(
    items: List[Tuple[str, str]],
    min_chars: int = MIN_LABEL_CHARS,
    max_chars: int = MAX_LABEL_CHARS,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """Drop too-short / too-long / non-textual labels and exact (rel_path, text) duplicates."""
    stats = {
        "input": len(items),
        "dropped_short": 0,
        "dropped_long": 0,
        "dropped_nontext": 0,
        "dropped_dup_pair": 0,
    }
    seen_pairs: Set[Tuple[str, str]] = set()
    kept: List[Tuple[str, str]] = []
    for rel_path, text in items:
        if len(text) < min_chars:
            stats["dropped_short"] += 1
            continue
        if len(text) > max_chars:
            stats["dropped_long"] += 1
            continue
        if not has_text_char(text):
            stats["dropped_nontext"] += 1
            continue
        key = (rel_path, text)
        if key in seen_pairs:
            stats["dropped_dup_pair"] += 1
            continue
        seen_pairs.add(key)
        kept.append((rel_path, text))
    stats["kept"] = len(kept)
    return kept, stats


def split_three(
    items: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    n = len(items)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    n_test = n - n_train - n_val
    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val :]
    return train_items, val_items, test_items


def write_split_labels(
    root: Path,
    train_pairs: List[Tuple[str, str]],
    val_pairs: List[Tuple[str, str]],
    test_pairs: List[Tuple[str, str]],
) -> None:
    def _write(path: Path, pairs: List[Tuple[str, str]]) -> None:
        with open(path, "w", encoding="utf-8") as fw:
            for rel, text in pairs:
                fw.write(f"{rel}\t{text}\n")

    _write(root / "rec_gt_train.txt", train_pairs)
    _write(root / "rec_gt_val.txt", val_pairs)
    _write(root / "rec_gt_test.txt", test_pairs)


STAGING_DIR_NAME = "_staging"


def process_items_to_staging(
    items: List[Tuple[str, str]],
    src_img_root: Path,
    bubble_root: Path,
    line_root: Path,
    cfg: BubbleLineConfig,
    seen_image_hashes: Set[str],
) -> Tuple[List[Tuple[str, str]], int]:
    """Process ``items`` and write successful pairs to ``<root>/_staging/``.

    Returns a list of ``(stage_name, text)`` in the same shuffled order as ``items`` and the
    number of samples dropped because their image bytes collided with a previously-seen MD5.
    """
    staging_b = bubble_root / STAGING_DIR_NAME
    staging_l = line_root / STAGING_DIR_NAME
    staging_b.mkdir(parents=True, exist_ok=True)
    staging_l.mkdir(parents=True, exist_ok=True)

    out_pairs: List[Tuple[str, str]] = []
    k = 0
    dup_content = 0
    for rel_path, text in tqdm(items, desc="Build staging"):
        src_path = src_img_root / rel_path
        try:
            raw_bytes = src_path.read_bytes()
        except Exception:
            continue
        digest = hashlib.md5(raw_bytes).hexdigest()
        if digest in seen_image_hashes:
            dup_content += 1
            continue
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        proc = bubble_to_line_image(img, cfg)
        k += 1
        stage_name = f"stage_{k:07d}.png"
        b_dst = staging_b / stage_name
        l_dst = staging_l / stage_name
        try:
            shutil.copy2(src_path, b_dst)
        except Exception:
            k -= 1
            continue
        if not cv2.imwrite(str(l_dst), proc["line"]):
            b_dst.unlink(missing_ok=True)
            k -= 1
            continue
        seen_image_hashes.add(digest)
        out_pairs.append((stage_name, text))
    return out_pairs, dup_content


def _cleanup_staging(bubble_root: Path, line_root: Path) -> None:
    for root in (bubble_root, line_root):
        staging = root / STAGING_DIR_NAME
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def finalize_split_from_staging(
    staging_pairs: List[Tuple[str, str]],
    bubble_root: Path,
    line_root: Path,
    max_samples: int,
) -> Dict[str, List[Tuple[str, str]]]:
    """Cap to ``max_samples``, split 80/10/10 and move staging files to final split dirs."""
    capped = staging_pairs[:max_samples]
    extras = staging_pairs[max_samples:]

    for stage_name, _ in extras:
        (bubble_root / STAGING_DIR_NAME / stage_name).unlink(missing_ok=True)
        (line_root / STAGING_DIR_NAME / stage_name).unlink(missing_ok=True)

    train_seg, val_seg, test_seg = split_three(capped)
    segments = {"train": train_seg, "val": val_seg, "test": test_seg}

    final_pairs: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": [], "test": []}
    for split_name, seg in segments.items():
        for i, (stage_name, text) in enumerate(seg, start=1):
            final_name = f"{split_name}_{i:06d}.png"
            for root in (bubble_root, line_root):
                src = root / STAGING_DIR_NAME / stage_name
                dst = root / split_name / final_name
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
            final_pairs[split_name].append((f"{split_name}/{final_name}", text))

    _cleanup_staging(bubble_root, line_root)
    return final_pairs


def build_bubble_and_line_datasets(
    src_label: Path,
    src_img_root: Path,
    bubble_root: Path,
    line_root: Path,
    cfg: BubbleLineConfig,
    max_samples: int,
    oversample_factor: float = OVERSAMPLE_FACTOR,
) -> Dict[str, object]:
    all_items = read_label_file(src_label)
    if not all_items:
        raise RuntimeError(f"No valid samples in {src_label}")

    filtered, filter_stats = filter_and_deduplicate(all_items, min_chars=MIN_LABEL_CHARS)
    if not filtered:
        raise RuntimeError("No samples left after filter + dedup.")

    rng = random.Random(SEED)
    items = filtered[:]
    rng.shuffle(items)

    oversample_k = min(int(max_samples * oversample_factor), len(items))
    selected = items[:oversample_k]

    _cleanup_staging(bubble_root, line_root)

    seen_hashes: Set[str] = set()
    staging_pairs, dup_content = process_items_to_staging(
        selected, src_img_root, bubble_root, line_root, cfg, seen_hashes
    )

    final_pairs = finalize_split_from_staging(
        staging_pairs, bubble_root, line_root, max_samples
    )

    write_split_labels(
        bubble_root, final_pairs["train"], final_pairs["val"], final_pairs["test"]
    )
    write_split_labels(
        line_root, final_pairs["train"], final_pairs["val"], final_pairs["test"]
    )

    return {
        "filter": filter_stats,
        "oversample_target": oversample_k,
        "staging_written": len(staging_pairs),
        "dropped_dup_image": dup_content,
        "capped_to": min(max_samples, len(staging_pairs)),
        "train_written": len(final_pairs["train"]),
        "val_written": len(final_pairs["val"]),
        "test_written": len(final_pairs["test"]),
    }


DATA_CFG = BubbleLineConfig()
build_stats = build_bubble_and_line_datasets(
    SRC_LABEL, SRC_IMAGE_ROOT, BUBBLE_ROOT, LINE_ROOT, DATA_CFG, MAX_SAMPLES
)
print(json.dumps(build_stats, indent=2, ensure_ascii=False))


# %% [markdown]
# ### Bước 9 — Từ điển ký tự `dict_japanese.txt` (PaddleOCR)
# 
# Gộp nhãn train + val + test của **một** bộ (hai bộ cùng nhãn), ghi `dict_japanese.txt` vào **cả** `bubble_dataset` và `line_dataset`.
# 

# %%
def build_dict_from_samples(pairs: List[Tuple[str, str]], dict_path: Path) -> int:
    charset = sorted({ch for _, t in pairs for ch in t})
    with open(dict_path, "w", encoding="utf-8") as f:
        for ch in charset:
            f.write(ch + "\n")
    return len(charset)


def all_label_pairs(root: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for name in ("rec_gt_train.txt", "rec_gt_val.txt", "rec_gt_test.txt"):
        pairs.extend(read_label_file(root / name))
    return pairs


pairs_all = all_label_pairs(BUBBLE_ROOT)
n_dict = build_dict_from_samples(pairs_all, BUBBLE_ROOT / "dict_japanese.txt")
build_dict_from_samples(pairs_all, LINE_ROOT / "dict_japanese.txt")
print("dict_japanese.txt classes:", n_dict)


# %% [markdown]
# ### Gợi ý cấu hình PaddleOCR `SimpleDataSet` sau này
# 
# - `data_dir`: đường dẫn tới `bubble_dataset` **hoặc** `line_dataset` (thư mục chứa `train/`, `val/`, `test/` và các `rec_gt_*.txt`).
# - Huấn luyện: `label_file_list: [<data_dir>/rec_gt_train.txt]`, ảnh trong nhãn là đường dẫn **tương đối** so với `data_dir` (ví dụ `train/train_000001.png`).
# - Validation: `rec_gt_val.txt`; benchmark hold-out: `rec_gt_test.txt`.
# - `character_dict_path`: `<data_dir>/dict_japanese.txt`.
# 

# %% [markdown]
# ### Bước 10 — Nén hai bộ dữ liệu để tải về
# 
# Tạo hai file `.tar.gz` trong `/kaggle/working/` (hiện trong Output của Kaggle).
# 

# %%
import tarfile
from datetime import datetime

PACK_DIR = Path("/kaggle/working")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

for stem, root in (
    ("bubble_dataset_60k", BUBBLE_ROOT),
    ("line_dataset_60k", LINE_ROOT),
):
    out_path = PACK_DIR / f"{stem}_{ts}.tar.gz"
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(root, arcname=root.name)
    mb = out_path.stat().st_size / (1024 ** 2)
    print("Archive:", out_path, f"({mb:.2f} MB)")



