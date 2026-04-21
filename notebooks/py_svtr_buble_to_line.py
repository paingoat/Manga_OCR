# %% [markdown]
# # SVTR Large (PaddleOCR) on Manga109 Rec with Bubble-to-Line Preprocess (45k cap)
# 
# Notebook này tập trung **thử nghiệm nhanh trên Kaggle H100** cho bài toán recognition:
# - Chỉ dùng **SVTR Large** (CTC)
# - Tiền xử lý bubble theo hướng **vertical projection -> line stitching**
# - Giới hạn dữ liệu train/val ở **45,000 ảnh**
# - Đánh giá bằng **CER** và **NED**
# 
# > Notebook chỉ cung cấp code để chạy trên Kaggle, không tự động chạy train.

# %% [markdown]
# ### Bước 0.1 - Cài Môi Trường Nền (Kaggle)
# Mục tiêu:
# - Đồng bộ môi trường cho PaddleOCR trên Kaggle H100.
# - Cài `paddlepaddle-gpu`, `paddlex`, `huggingface_hub` và các thư viện tiện ích.
# 
# Ghi chú:
# - Cell này chỉ cần chạy 1 lần mỗi phiên runtime mới.
# - Nếu Kaggle báo cần restart runtime sau khi cài package, hãy restart rồi chạy lại từ đầu notebook.

# %%
# =========================
# 0.1) Install base dependencies
# =========================
import sys
import subprocess

# Tránh conflict từ stack torch preinstalled (notebook này chỉ dùng Paddle)
subprocess.run([
    sys.executable, '-m', 'pip', 'uninstall', '-y',
    'torch', 'torchvision', 'torchaudio', 'triton',
    'fastai', 'fastcore', 'fastdownload', 'nbdev'
], check=False)

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'pip', 'setuptools', 'wheel'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'paddlepaddle-gpu==3.0.0', '-i', 'https://www.paddlepaddle.org.cn/packages/stable/cu126/'])

import paddle
print('Paddle version:', paddle.__version__)
print('CUDA available:', paddle.device.is_compiled_with_cuda())
print('GPU count:', paddle.device.cuda.device_count())
for i in range(paddle.device.cuda.device_count()):
    print(f'GPU {i}:', paddle.device.cuda.get_device_name(i))

# %% [markdown]
# ### Bước 0.2 - Clone PaddleOCR và Cài Requirements
# Mục tiêu:
# - Tạo thư mục PaddleOCR trong `/kaggle/working`.
# - Cài các dependency chuẩn từ `requirements.txt` của PaddleOCR.
# 
# Output mong đợi:
# - In ra đường dẫn repo và xác nhận đã sẵn sàng train/infer.

# %%
# =========================
# 0.2) Clone PaddleOCR repo
# =========================
import os
from pathlib import Path

PADDLEOCR_DIR_BOOTSTRAP = Path('/kaggle/working/PaddleOCR')
if not PADDLEOCR_DIR_BOOTSTRAP.is_dir():
    !git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git {PADDLEOCR_DIR_BOOTSTRAP}

!pip install -q -r {PADDLEOCR_DIR_BOOTSTRAP}/requirements.txt
print('PaddleOCR ready:', PADDLEOCR_DIR_BOOTSTRAP)

# %% [markdown]
# ### Bước 0.3 - Đăng Nhập Hugging Face Token (Tuỳ Chọn)
# Mục tiêu:
# - Đăng nhập Hugging Face để tải dataset/model private nếu cần.
# 
# Cách dùng trên Kaggle:
# - Vào `Add-ons -> Secrets`, tạo secret tên `HF_TOKEN`.
# - Chạy cell bên dưới để login tự động.

# %%
# =========================
# 0.3) HuggingFace login via Kaggle Secrets
# =========================
import subprocess
import sys

# Bao dam huggingface_hub da co
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'])

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("HF_TOKEN")

if secret_value_0 and str(secret_value_0).strip():
    login(str(secret_value_0).strip())
    print('HF login success via kaggle_secrets.HF_TOKEN')
else:
    print('HF_TOKEN secret is empty or missing.')

# %% [markdown]
# ### Bước 1 - Environment Setup
# Cell này import thư viện, khai báo đường dẫn dữ liệu/output và các tham số nền tảng (`seed`, `MAX_SAMPLES`, `VAL_RATIO`).

# %%
# =========================
# 1) Environment setup
# =========================
import os
import re
import cv2
import yaml
import json
import math
import shutil
import random
import unicodedata
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

SEED = 2026
random.seed(SEED)
np.random.seed(SEED)

# ---- Kaggle-friendly defaults (chỉnh theo dataset của bạn) ----
PADDLEOCR_DIR = Path('/kaggle/working/PaddleOCR')
DATA_ROOT_DEFAULT = Path('/kaggle/working/train_data/rec')
OUTPUT_ROOT = Path('/kaggle/working/exp_svtr_45k_bubbleline')

# Neu da chay cac buoc download/extract o tren, DATA_ROOT se duoc override tu bien global DATA_ROOT_OVERRIDE
DATA_ROOT = Path(globals().get('DATA_ROOT_OVERRIDE', str(DATA_ROOT_DEFAULT)))

SRC_LABEL = DATA_ROOT / 'rec_gt_train.txt'     # file gốc (tab: rel_path\ttext)
SRC_IMAGE_ROOT = DATA_ROOT                      # root để resolve rel_path

WORK_DATA = OUTPUT_ROOT / 'data'
TRAIN_DIR = WORK_DATA / 'train'
VAL_DIR = WORK_DATA / 'val'
TRAIN_LABEL = WORK_DATA / 'rec_gt_train.txt'
VAL_LABEL = WORK_DATA / 'rec_gt_val.txt'
DICT_PATH = WORK_DATA / 'dict_japanese.txt'

CONFIG_PATH = OUTPUT_ROOT / 'svtr_45k_bubbleline.yml'
TRAIN_LOG = OUTPUT_ROOT / 'train.log'
EVAL_JSON = OUTPUT_ROOT / 'eval_cer_ned.json'
REC_IMAGE_SHAPE_STR = '3,32,320'  # Keep train/infer shape consistent for SVTR

MAX_SAMPLES = 45_000
VAL_RATIO = 0.08

for p in [OUTPUT_ROOT, WORK_DATA, TRAIN_DIR, VAL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print('PADDLEOCR_DIR =', PADDLEOCR_DIR)
print('SRC_LABEL     =', SRC_LABEL)
print('OUTPUT_ROOT   =', OUTPUT_ROOT)

# %% [markdown]
# ### Bước 2 - (Tương thích H100) Tải Dataset Từ Hugging Face
# Mục tiêu:
# - Tải snapshot Manga109-s từ Hugging Face (nếu bạn chưa mount sẵn dữ liệu rec).
# - Không ghi đè dữ liệu nếu file zip đã tồn tại.
# 
# Ghi chú:
# - Cần chạy Bước 0.3 trước để có `HF_TOKEN`.
# - Nếu bạn đã có `rec_gt_train.txt` ở `/kaggle/working/train_data/rec` thì có thể bỏ qua bước này.

# %%
# =========================
# 2) Download Manga109-s from HF (optional)
# =========================
import zipfile
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    raise RuntimeError('huggingface_hub is required. Run Step 0.1 first.') from e

WORK_DIR = Path('/kaggle/working')
HF_DOWNLOAD_DIR = WORK_DIR / 'manga109s_dataset'
EXTRACT_DIR = WORK_DIR / 'manga109_extracted'
DATASET_ROOT = EXTRACT_DIR / 'Manga109s_released_2023_12_07'
ZIP_PATH = HF_DOWNLOAD_DIR / 'Manga109s_released_2023_12_07.zip'

HF_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    if 'secret_value_0' not in globals() or not str(secret_value_0).strip():
        raise RuntimeError('HF token not found. Run Step 0.3 first or skip this step if data already exists.')

    snapshot_download(
        repo_id='hal-utokyo/Manga109-s',
        repo_type='dataset',
        local_dir=str(HF_DOWNLOAD_DIR),
        token=str(secret_value_0).strip(),
    )

print('HF download dir:', HF_DOWNLOAD_DIR)
print('Zip exists:', ZIP_PATH.exists())

# %% [markdown]
# ### Bước 3 - (Tương thích H100) Giải Nén Dataset
# Mục tiêu:
# - Giải nén file zip Manga109-s vào thư mục làm việc.
# - Chỉ giải nén khi chưa có dữ liệu đã extract.
# 
# Output mong đợi:
# - In ra `DATASET_ROOT` và liệt kê một phần nội dung để kiểm tra.

# %%
# =========================
# 3) Extract dataset zip (optional)
# =========================
if 'ZIP_PATH' not in globals():
    raise RuntimeError('Run Step 2 first to define ZIP_PATH.')

if ZIP_PATH.exists() and (not DATASET_ROOT.exists() or not any(DATASET_ROOT.iterdir())):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_DIR)

print('DATASET_ROOT:', DATASET_ROOT)
if DATASET_ROOT.exists():
    print('Top-level content:', sorted([p.name for p in DATASET_ROOT.iterdir()])[:20])
else:
    print('DATASET_ROOT not found. You can skip Steps 2-3 if using pre-prepared rec data.')

# %% [markdown]
# ### Bước 4 - Chuẩn Hóa Nguồn Dữ Liệu `rec` Trước Khi Build 45k
# Mục tiêu:
# - Ưu tiên dùng nguồn `rec` đã chuẩn bị sẵn nếu có.
# - Nếu tìm thấy nhánh `rec` trong dữ liệu đã extract, tự động dùng nhánh đó.
# - Kiểm tra tồn tại của `rec_gt_train.txt` và in vài dòng mẫu để sanity-check.

# %%
# =========================
# 4) Resolve DATA_ROOT and sanity-check rec labels
# =========================
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image


def _find_existing_rec_root() -> Path:
    candidate_roots = [
        Path('/kaggle/working/train_data/rec'),
        Path('/kaggle/working/manga109/rec'),
    ]

    if 'DATASET_ROOT' in globals() and Path(DATASET_ROOT).exists():
        candidate_roots.extend([
            Path(DATASET_ROOT) / 'rec',
            Path(DATASET_ROOT) / 'train_data' / 'rec',
            Path(DATASET_ROOT),
        ])

    kaggle_working = Path('/kaggle/working')
    if kaggle_working.exists():
        for rec_label in kaggle_working.rglob('rec_gt_train.txt'):
            candidate_roots.append(rec_label.parent)

    seen = set()
    uniq_candidates = []
    for p in candidate_roots:
        key = str(p)
        if key not in seen:
            uniq_candidates.append(p)
            seen.add(key)

    checked = []
    for c in uniq_candidates:
        probe = c / 'rec_gt_train.txt'
        checked.append(str(probe))
        if probe.exists():
            return c

    print('Checked candidates:')
    for p in checked[:30]:
        print(' -', p)
    if len(checked) > 30:
        print(f' ... and {len(checked)-30} more')
    return None


def _build_rec_from_manga109s(dataset_root: Path, out_root: Path, max_samples: int = 120_000) -> Path:
    annotations_dir = dataset_root / 'annotations'
    images_dir = dataset_root / 'images'

    if not annotations_dir.exists() or not images_dir.exists():
        raise FileNotFoundError(
            f'Cannot build rec source: missing annotations/images under {dataset_root}'
        )

    out_root.mkdir(parents=True, exist_ok=True)
    out_train = out_root / 'train'
    out_train.mkdir(parents=True, exist_ok=True)
    out_label = out_root / 'rec_gt_train.txt'

    xml_files = sorted([p for p in annotations_dir.glob('*.xml')])
    if not xml_files:
        raise RuntimeError(f'No XML annotation files found in {annotations_dir}')

    print(f'Building rec source from raw Manga109-s: {len(xml_files)} books')
    num_written = 0

    with open(out_label, 'w', encoding='utf-8') as fw:
        for xml_path in tqdm(xml_files, desc='Parse books'):
            book_title = xml_path.stem
            try:
                root = ET.parse(xml_path).getroot()
            except Exception:
                continue

            for page in root.findall('.//page'):
                page_idx = page.get('index')
                if page_idx is None:
                    continue
                img_path = images_dir / book_title / f'{int(page_idx):03d}.jpg'
                if not img_path.exists():
                    continue

                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception:
                    continue

                for text_obj in page.findall('text'):
                    try:
                        xmin = int(text_obj.get('xmin'))
                        ymin = int(text_obj.get('ymin'))
                        xmax = int(text_obj.get('xmax'))
                        ymax = int(text_obj.get('ymax'))
                        label = (text_obj.text or '').strip().replace('\n', '').replace('\t', ' ')

                        if not label:
                            continue
                        if (xmax - xmin) < 12 or (ymax - ymin) < 12:
                            continue

                        crop = img.crop((xmin, ymin, xmax, ymax))
                        num_written += 1
                        out_name = f'word_{num_written:07d}.png'
                        crop.save(out_train / out_name)
                        fw.write(f'train/{out_name}\t{label}\n')

                        if max_samples and num_written >= max_samples:
                            print(f'Reached max_samples={max_samples}')
                            return out_root
                    except Exception:
                        continue

    return out_root


resolved = _find_existing_rec_root()

if resolved is None:
    if 'DATASET_ROOT' in globals() and Path(DATASET_ROOT).exists():
        auto_out = Path('/kaggle/working/train_data/rec')
        print('\nrec_gt_train.txt not found. Fallback: auto-build rec source from extracted Manga109-s...')
        resolved = _build_rec_from_manga109s(Path(DATASET_ROOT), auto_out, max_samples=120_000)
    else:
        raise FileNotFoundError(
            'Khong tim thay rec_gt_train.txt va DATASET_ROOT cung khong ton tai. '
            'Hay chay Steps 2-3 de tai/extract dataset raw truoc.'
        )

DATA_ROOT_OVERRIDE = str(resolved)
print('Resolved DATA_ROOT_OVERRIDE =', DATA_ROOT_OVERRIDE)

label_path = Path(DATA_ROOT_OVERRIDE) / 'rec_gt_train.txt'
if not label_path.exists():
    raise FileNotFoundError(f'Label file missing after resolve/build: {label_path}')

print('Label path:', label_path)
print('\nSample label lines:')
with open(label_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        print(line.rstrip('\n'))
        if i >= 4:
            break

# %% [markdown]
# ### Bước 5 - Định Nghĩa Pipeline Bubble-to-Line
# Cell này chứa toàn bộ hàm preprocess: nhị phân hóa, lọc nhiễu, tách cột bằng projection, xoay cột và ghép thành line ngang.

# %%
# =========================
# 2) Bubble -> line preprocess (clean version)
# =========================
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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', '', text)
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


def vertical_projection_intervals(mask: np.ndarray, cfg: BubbleLineConfig) -> List[Tuple[int, int]]:
    profile = mask.sum(axis=0).astype(np.float32)
    active = profile[profile > 0]
    if active.size == 0:
        return [(0, mask.shape[1])]

    th = float(np.percentile(active, cfg.gap_percentile) * cfg.gap_scale)
    gap_mask = profile <= th
    gaps = find_gap_runs(gap_mask, cfg.min_gap_width)

    cuts = [0]
    for g0, g1 in gaps:
        cuts.append((g0 + g1) // 2)
    cuts.append(mask.shape[1])
    cuts = sorted(set(cuts))

    intervals: List[Tuple[int, int]] = []
    for i in range(len(cuts) - 1):
        x0, x1 = cuts[i], cuts[i + 1]
        if x1 - x0 >= cfg.min_col_width:
            intervals.append((x0, x1))

    if not intervals:
        intervals = [(0, mask.shape[1])]
    return intervals


def crop_fg_y(mask: np.ndarray, x0: int, x1: int) -> Optional[Tuple[int, int]]:
    col = mask[:, x0:x1]
    ys = np.where(col.sum(axis=1) > 0)[0]
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return y0, y1


def rotate_and_resize_column(col_img: np.ndarray, target_h: int) -> np.ndarray:
    # Vertical JP column -> horizontal by CCW rotate
    rot = cv2.rotate(col_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = rot.shape[:2]
    if h <= 0 or w <= 0:
        return rot
    new_w = max(1, int(round(w * (target_h / h))))
    return cv2.resize(rot, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def bubble_to_line_image(bgr: np.ndarray, cfg: BubbleLineConfig) -> Dict[str, np.ndarray]:
    bw = preprocess_binarize(bgr)
    cleaned = remove_small_components(bw, cfg.min_component_area)
    intervals = vertical_projection_intervals(cleaned, cfg)

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
        # fallback: resize whole bubble
        h, w = bgr.shape[:2]
        new_w = max(1, int(round(w * (cfg.target_h / max(1, h)))))
        line = cv2.resize(bgr, (new_w, cfg.target_h), interpolation=cv2.INTER_LINEAR)
    else:
        spacer = np.full((cfg.target_h, cfg.spacer_px, 3), 255, dtype=np.uint8)
        parts = []
        for i, c in enumerate(columns):
            parts.append(c)
            if i < len(columns) - 1:
                parts.append(spacer)
        line = np.concatenate(parts, axis=1)

    debug = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    for x0, x1 in intervals:
        cv2.rectangle(debug, (x0, 0), (x1, debug.shape[0] - 1), (0, 255, 255), 1)

    return {
        'binary': bw,
        'cleaned': cleaned,
        'debug_projection': debug,
        'line': line,
    }

# %% [markdown]
# ### Bước 6 - Trực Quan Hóa Nhanh
# Cell này hiển thị ảnh đầu vào, mask nhị phân, vùng tách cột và ảnh line cuối để kiểm tra chất lượng preprocess.

# %%
# =========================
# 3) Small visualization check (optional)
# =========================

def show_preview(src_path: Path, cfg: BubbleLineConfig) -> None:
    img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {src_path}')

    out = bubble_to_line_image(img, cfg)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Input bubble')
    axes[1].imshow(out['binary'], cmap='gray')
    axes[1].set_title('Binary')
    axes[2].imshow(cv2.cvtColor(out['debug_projection'], cv2.COLOR_BGR2RGB))
    axes[2].set_title('Projection intervals')
    axes[3].imshow(cv2.cvtColor(out['line'], cv2.COLOR_BGR2RGB))
    axes[3].set_title('Stitched horizontal line')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()

# Example usage:
from pathlib import Path

# Ưu tiên root đã resolve ở bước 4
root = Path(globals().get("DATA_ROOT_OVERRIDE", str(DATA_ROOT)))
label_path = root / "rec_gt_train.txt"

with open(label_path, "r", encoding="utf-8") as f:
    rel_path, _ = f.readline().strip().split("\t", 1)

preview_img = root / rel_path
print("Preview path:", preview_img, "exists:", preview_img.exists())

show_preview(preview_img, BubbleLineConfig())

# %% [markdown]
# ### Bước 7 - Build Dataset 45k
# Cell này đọc label gốc, giới hạn tối đa 45k mẫu, chạy preprocess bubble->line, ghi ảnh mới và file label train/val.

# %%
# =========================
# 4) Build 45k dataset with preprocess
# =========================

def read_label_file(label_path: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' not in line:
                continue
            rel_path, text = line.split('\t', 1)
            text = normalize_text(text)
            if not text:
                continue
            items.append((rel_path, text))
    return items


def build_dict_from_samples(samples: List[Tuple[str, str]], dict_path: Path) -> int:
    charset = sorted({ch for _, t in samples for ch in t})
    with open(dict_path, 'w', encoding='utf-8') as f:
        for ch in charset:
            f.write(ch + '\n')
    return len(charset)


def build_dataset_45k(
    src_label: Path,
    src_img_root: Path,
    train_dir: Path,
    val_dir: Path,
    train_label_out: Path,
    val_label_out: Path,
    cfg: BubbleLineConfig,
    max_samples: int = 45_000,
    val_ratio: float = 0.08,
) -> Dict[str, int]:
    all_items = read_label_file(src_label)
    if not all_items:
        raise RuntimeError(f'No valid samples in {src_label}')

    random.shuffle(all_items)
    selected = all_items[: min(max_samples, len(all_items))]

    n_val = max(1, int(len(selected) * val_ratio))
    val_items = selected[:n_val]
    train_items = selected[n_val:]

    def _process_split(split_items: List[Tuple[str, str]], out_dir: Path, out_label: Path, split_name: str) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(out_label, 'w', encoding='utf-8') as fw:
            for idx, (rel_path, text) in enumerate(tqdm(split_items, desc=f'Preprocess {split_name}', leave=False), start=1):
                src_img = src_img_root / rel_path
                img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                proc = bubble_to_line_image(img, cfg)
                out_name = f'{split_name}_{idx:06d}.png'
                out_path = out_dir / out_name
                ok = cv2.imwrite(str(out_path), proc['line'])
                if not ok:
                    continue

                fw.write(f'{split_name}/{out_name}\t{text}\n')
                written += 1
        return written

    n_train = _process_split(train_items, train_dir, train_label_out, 'train')
    n_val_w = _process_split(val_items, val_dir, val_label_out, 'val')

    return {
        'selected': len(selected),
        'train': n_train,
        'val': n_val_w,
    }


build_stats = build_dataset_45k(
    src_label=SRC_LABEL,
    src_img_root=SRC_IMAGE_ROOT,
    train_dir=TRAIN_DIR,
    val_dir=VAL_DIR,
    train_label_out=TRAIN_LABEL,
    val_label_out=VAL_LABEL,
    cfg=BubbleLineConfig(),
    max_samples=MAX_SAMPLES,
    val_ratio=VAL_RATIO,
)

n_classes = build_dict_from_samples(read_label_file(TRAIN_LABEL) + read_label_file(VAL_LABEL), DICT_PATH)
print('Build done:', build_stats)
print('Dictionary classes:', n_classes)
print('TRAIN_LABEL:', TRAIN_LABEL)
print('VAL_LABEL  :', VAL_LABEL)

# %% [markdown]
# ### Bước 8 - Tải Pretrained SVTR Large (Chinese)
# Cell này khai báo checkpoint pretrained khởi tạo theo bản tiếng Trung (`rec_svtr_large_none_ctc_ch_train`) để fine-tune cho tiếng Nhật trên Kaggle H100.

# %%
# =========================
# 5) Download pretrained SVTR Large (optional)
# =========================
PRETRAINED_DIR = OUTPUT_ROOT / 'pretrained'
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

# Dùng pretrained Chinese SVTR Large theo yêu cầu thử nghiệm
PRETRAINED_URL = 'https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_large_none_ctc_ch_train.tar'
PRETRAINED_TAR = PRETRAINED_DIR / 'rec_svtr_large_none_ctc_ch_train.tar'
PRETRAINED_ROOT = PRETRAINED_DIR / 'rec_svtr_large_none_ctc_ch_train'
PRETRAINED_PATH = PRETRAINED_ROOT / 'best_accuracy'

print('Pretrained URL :', PRETRAINED_URL)
print('Pretrained path:', PRETRAINED_PATH)

# Uncomment on Kaggle when running:
!wget -q --show-progress -O {PRETRAINED_TAR} {PRETRAINED_URL}
!tar -xf {PRETRAINED_TAR} -C {PRETRAINED_DIR}

# %% [markdown]
# ### Bước 9 - Sinh YAML Config Huấn Luyện (SVTR)
# Cell này tạo file config PaddleOCR cho SVTR (CTC), dùng dữ liệu đã preprocess và dictionary đã build, tối ưu tham số cho Kaggle H100.

# %%
# =========================
# 6) Build PaddleOCR train config (SVTR + Kaggle H100 profile)
# =========================

def count_lines(path: Path) -> int:
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

N_GPU = 1
BATCH_PER_CARD = int(globals().get('BATCH_PER_CARD_OVERRIDE', 256))
NUM_WORKERS = int(globals().get('NUM_WORKERS_OVERRIDE', 4))
EVAL_BATCH_PER_CARD = int(globals().get('EVAL_BATCH_PER_CARD_OVERRIDE', BATCH_PER_CARD))
REC_IMAGE_SHAPE = [int(x) for x in REC_IMAGE_SHAPE_STR.split(',')]
n_train = count_lines(TRAIN_LABEL)
steps_per_epoch = max(1, n_train // (BATCH_PER_CARD * N_GPU))
eval_step = max(500, steps_per_epoch * 2)

pretrained_params_file = PRETRAINED_PATH.with_suffix('.pdparams')
pretrained_exists = pretrained_params_file.exists()
pretrained_for_train = str(PRETRAINED_PATH) if pretrained_exists else ''
if pretrained_exists:
    print('Using pretrained checkpoint:', PRETRAINED_PATH)
    print('Found:', pretrained_params_file)
else:
    print('Pretrained .pdparams not found -> training from scratch')
    print('Missing:', pretrained_params_file)

def _load_pretrained_svtr_architecture(pretrained_root: Path) -> Dict:
    # Reuse official Architecture from pretrained package when available
    for yml_path in sorted(pretrained_root.glob('*.yml')):
        try:
            with open(yml_path, 'r', encoding='utf-8') as f:
                yml_cfg = yaml.safe_load(f)
            arch = yml_cfg.get('Architecture') if isinstance(yml_cfg, dict) else None
            if isinstance(arch, dict) and arch.get('algorithm') == 'SVTR':
                print('Use SVTR architecture from pretrained yaml:', yml_path.name)
                return arch
        except Exception as exc:
            print(f'Skip invalid yaml {yml_path.name}: {exc}')

    print('Fallback to built-in SVTR architecture defaults')
    return {
        'model_type': 'rec',
        'algorithm': 'SVTR',
        'Transform': None,
        'Backbone': {
            'name': 'SVTRNet',
            'img_size': [32, 320],
            'out_channels': 192,
        },
        'Neck': {'name': 'SequenceEncoder', 'encoder_type': 'reshape'},
        'Head': {'name': 'CTCHead', 'fc_decay': 1e-5},
    }

architecture_cfg = _load_pretrained_svtr_architecture(PRETRAINED_ROOT)

cfg = {
    'Global': {
        'use_gpu': True,
        'epoch_num': 80,
        'log_smooth_window': 20,
        'print_batch_step': 20,
        'save_model_dir': str(OUTPUT_ROOT / 'output'),
        'save_epoch_step': 2,
        'eval_batch_step': [0, eval_step],
        'cal_metric_during_train': True,
        'pretrained_model': pretrained_for_train,
        'character_dict_path': str(DICT_PATH),
        'max_text_length': 80,
        'infer_mode': False,
        'use_space_char': False,
        'save_res_path': str(OUTPUT_ROOT / 'predicts.txt'),
        'seed': SEED,
    },
    'Optimizer': {
        'name': 'Adam',
        'beta1': 0.9,
        'beta2': 0.999,
        'lr': {'name': 'Cosine', 'learning_rate': 4e-4, 'warmup_epoch': 2},
        'regularizer': {'name': 'L2', 'factor': 1e-5},
    },
    'Architecture': architecture_cfg,
    'Loss': {'name': 'CTCLoss'},
    'PostProcess': {'name': 'CTCLabelDecode'},
    'Metric': {'name': 'RecMetric', 'main_indicator': 'acc'},
    'Train': {
        'dataset': {
            'name': 'SimpleDataSet',
            'data_dir': str(WORK_DATA),
            'label_file_list': [str(TRAIN_LABEL)],
            'transforms': [
                {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                {'RecAug': None},
                {'CTCLabelEncode': None},
                {'RecResizeImg': {'image_shape': REC_IMAGE_SHAPE}},
                {'KeepKeys': {'keep_keys': ['image', 'label', 'length']}},
            ],
        },
        'loader': {
            'shuffle': True,
            'batch_size_per_card': BATCH_PER_CARD,
            'drop_last': True,
            'num_workers': NUM_WORKERS,
            'use_shared_memory': True,
        },
    },
    'Eval': {
        'dataset': {
            'name': 'SimpleDataSet',
            'data_dir': str(WORK_DATA),
            'label_file_list': [str(VAL_LABEL)],
            'transforms': [
                {'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}},
                {'CTCLabelEncode': None},
                {'RecResizeImg': {'image_shape': REC_IMAGE_SHAPE}},
                {'KeepKeys': {'keep_keys': ['image', 'label', 'length']}},
            ],
        },
        'loader': {
            'shuffle': False,
            'drop_last': False,
            'batch_size_per_card': EVAL_BATCH_PER_CARD,
            'num_workers': NUM_WORKERS,
            'use_shared_memory': True,
        },
    },
}

with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print('Config saved to:', CONFIG_PATH)
print('Train samples:', n_train)
print('Val samples  :', count_lines(VAL_LABEL))
print('BATCH_PER_CARD:', BATCH_PER_CARD)
print('NUM_WORKERS   :', NUM_WORKERS)
print('REC_IMAGE_SHAPE:', REC_IMAGE_SHAPE)

# %% [markdown]
# ### Bước 10 - Train Với Progress Bar
# Cell này đóng gói lệnh train PaddleOCR và parse log để hiển thị tiến độ epoch ngay trong notebook.

# %%
# =========================
# 7) Train command with notebook progress bar parser
# =========================

def run_train_with_progress(config_path: Path, log_path: Path, paddleocr_dir: Path) -> int:
    """
    Chạy train.py và parse stdout để hiện progress bar theo epoch.
    Không gọi hàm này nếu bạn chỉ muốn chuẩn bị notebook.
    """
    train_script = paddleocr_dir / 'tools' / 'train.py'
    if not train_script.exists():
        raise FileNotFoundError(f'Cannot find train script: {train_script}')

    cmd = ['python', str(train_script), '-c', str(config_path)]
    print('Running:', ' '.join(cmd))

    total_epoch = None
    pbar = None

    with open(log_path, 'w', encoding='utf-8') as fw:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=str(paddleocr_dir),
        )

        for line in proc.stdout:
            fw.write(line)

            # Ví dụ log có thể chứa: "[3/80]" hoặc "epoch: [3/80]"
            m = re.search(r'\[(\d+)\s*/\s*(\d+)\]', line)
            if m:
                cur_ep = int(m.group(1))
                max_ep = int(m.group(2))
                if total_epoch is None:
                    total_epoch = max_ep
                    pbar = tqdm(total=total_epoch, desc='Training epochs', unit='ep')
                if pbar is not None:
                    pbar.n = min(cur_ep, total_epoch)
                    pbar.refresh()

            if 'best metric' in line.lower() or 'eval' in line.lower():
                # giữ một số line quan trọng hiển thị trực tiếp
                print(line.strip())

        rc = proc.wait()

    if pbar is not None:
        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

    print('Train finished with code:', rc)
    print('Train log:', log_path)

    if rc != 0:
        print('\n=== Last 120 lines of train log ===')
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as fr:
                lines = fr.readlines()
            for line in lines[-120:]:
                print(line.rstrip('\n'))
        except Exception as e:
            print('Could not read train log tail:', e)

    return rc

# Uncomment để chạy train trên Kaggle:
rc = run_train_with_progress(CONFIG_PATH, TRAIN_LOG, PADDLEOCR_DIR)
if rc != 0:
    raise RuntimeError('Training failed')

# %% [markdown]
# ### Bước 11 - Inference Trên Validation Set
# Cell này chuẩn bị danh sách ảnh validation và gọi `predict_rec.py` để sinh dự đoán phục vụ tính metric.

# %%
from pathlib import Path
import yaml

CONFIG_PATH = Path("/kaggle/working/exp_svtr_45k_bubbleline/svtr_45k_bubbleline.yml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def patch_transforms(transforms):
    if not transforms:
        return
    for i, item in enumerate(transforms):
        if not isinstance(item, dict):
            continue
        if "RecResizeImg" in item:
            old = item.pop("RecResizeImg")
            shape = old.get("image_shape")
            item["SVTRRecResizeImg"] = {"image_shape": shape, "padding": False}

for split in ("Train", "Eval"):
    ds = cfg.get(split, {}).get("dataset", {})
    patch_transforms(ds.get("transforms"))

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("Patched:", CONFIG_PATH)

# %%
# =========================
# 8) Inference on val set (for CER + NED)
# =========================

def read_gt_map(label_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            rel, txt = line.split('\t', 1)
            rows.append((rel, normalize_text(txt)))
    return rows


def export_inference_model_if_needed(config_path: Path, paddleocr_dir: Path, checkpoint_prefix: Path, infer_dir: Path) -> None:
    """
    Export checkpoint train -> inference model.
    Voi PaddleOCR moi, thuong co inference.json + inference.pdiparams (+ inference.yml).
    """
    export_script = paddleocr_dir / 'tools' / 'export_model.py'
    if not export_script.exists():
        raise FileNotFoundError(f'Cannot find export script: {export_script}')

    infer_json = infer_dir / 'inference.json'
    infer_pdiparams = infer_dir / 'inference.pdiparams'
    infer_pdmodel = infer_dir / 'inference.pdmodel'

    # Chap nhan ca 2 dinh dang cu/moi
    if (infer_json.exists() and infer_pdiparams.exists()) or (infer_pdmodel.exists() and infer_pdiparams.exists()):
        print('Inference model exists, skip export:', infer_dir)
        return

    infer_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        'python', str(export_script),
        '-c', str(config_path),
        '-o',
        f'Global.pretrained_model={checkpoint_prefix}',
        f'Global.save_inference_dir={infer_dir}',
    ]
    print('Export inference model...')
    subprocess.run(cmd, cwd=str(paddleocr_dir), check=True)


def run_predict_rec_on_val_dir(
    paddleocr_dir: Path,
    val_dir: Path,
    infer_model_dir: Path,
    out_log_path: Path,
    out_res_path: Path,
) -> None:
    infer_script = paddleocr_dir / 'tools' / 'infer' / 'predict_rec.py'
    if not infer_script.exists():
        raise FileNotFoundError(f'Cannot find infer script: {infer_script}')

    base_cmd = [
        'python', str(infer_script),
        '--rec_model_dir', str(infer_model_dir),
        '--image_dir', str(val_dir),
        '--rec_algorithm', 'SVTR',
        '--rec_image_shape', REC_IMAGE_SHAPE_STR,
        '--rec_char_dict_path', str(DICT_PATH),
        '--use_space_char', 'False',
    ]

    cmd_with_save = base_cmd + ['--save_res_path', str(out_res_path)]

    # Thu voi --save_res_path truoc (neu version PaddleOCR ho tro)
    with open(out_log_path, 'w', encoding='utf-8') as fw:
        rc = subprocess.run(cmd_with_save, cwd=str(paddleocr_dir), stdout=fw, stderr=subprocess.STDOUT).returncode

    if rc == 0:
        print('Inference done with --save_res_path:', out_res_path)
        return

    # Fallback: mot so version predict_rec.py khong co --save_res_path
    with open(out_log_path, 'r', encoding='utf-8', errors='ignore') as fr:
        log_text = fr.read()

    if ('unrecognized arguments: --save_res_path' in log_text) or ('error: unrecognized arguments' in log_text):
        print('predict_rec.py does not support --save_res_path, fallback to stdout log parsing')
        with open(out_log_path, 'w', encoding='utf-8') as fw:
            rc2 = subprocess.run(base_cmd, cwd=str(paddleocr_dir), stdout=fw, stderr=subprocess.STDOUT).returncode
        if rc2 == 0:
            print('Inference done (stdout mode). Log saved to:', out_log_path)
            return

    # Neu van fail, in tail log de de debug
    print('\n=== Inference log tail ===')
    with open(out_log_path, 'r', encoding='utf-8', errors='ignore') as fr:
        lines = fr.readlines()
    for line in lines[-120:]:
        print(line.rstrip('\n'))

    raise RuntimeError(f'Inference failed. See log: {out_log_path}')


BEST_CKPT_PREFIX = OUTPUT_ROOT / 'output' / 'best_accuracy'
INFER_MODEL_DIR = OUTPUT_ROOT / 'output' / 'inference'
VAL_IMAGE_DIR = WORK_DATA / 'val'
PRED_RAW = OUTPUT_ROOT / 'val_predict_raw.txt'
PRED_RES = OUTPUT_ROOT / 'val_predict_results.txt'

print('BEST_CKPT_PREFIX:', BEST_CKPT_PREFIX)
print('INFER_MODEL_DIR :', INFER_MODEL_DIR)
print('VAL_IMAGE_DIR   :', VAL_IMAGE_DIR)
print('PRED_RAW(LOG)   :', PRED_RAW)
print('PRED_RES(RESULT):', PRED_RES)

# Uncomment de chay sau khi train:
export_inference_model_if_needed(CONFIG_PATH, PADDLEOCR_DIR, BEST_CKPT_PREFIX, INFER_MODEL_DIR)
run_predict_rec_on_val_dir(PADDLEOCR_DIR, VAL_IMAGE_DIR, INFER_MODEL_DIR, PRED_RAW, PRED_RES)

# %% [markdown]
# ### Bước 12 - Tính CER và NED
# Cell này ưu tiên đọc file kết quả từ `--save_res_path` (ổn định hơn parse stdout), fallback sang log raw nếu cần; sau đó chuẩn hóa text, đối sánh theo path hoặc basename và tính `CER`, `NED`, `Exact Match`.

# %%
# =========================
# 9) CER / NED computation
# =========================

def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                prev[j] + 1,      # delete
                cur[j - 1] + 1,   # insert
                prev[j - 1] + cost,
            ))
        prev = cur
    return prev[-1]


def parse_predict_output(raw_path: Path) -> Dict[str, str]:
    """
    Parse output tu predict_rec.py (nhieu format), bao gom:
    - Predicts of /path/img.png:('text', 0.99)
    - /path/img.png\t('text', 0.99)
    - [time] ppocr INFO: /path/img.png\t('text', 0.99)
    - [time] ppocr INFO: result: ('text', 0.99)\t/path/img.png
    """
    import ast

    pred_map: Dict[str, str] = {}

    # Format moi thuong gap trong log cua ban: "Predicts of <path>:('text', score)"
    p_predicts_of = re.compile(
        r"Predicts\s+of\s+(?P<path>.+?\.(?:png|jpg|jpeg|bmp|webp|gif)):(?P<tpl>\(.*\))$",
        re.IGNORECASE,
    )

    # Format tab: <path>\t('text', score)
    p_path_then_tuple = re.compile(
        r"(?P<path>[^\t]+\.(?:png|jpg|jpeg|bmp|webp|gif))\t(?P<tpl>\(.*\))$",
        re.IGNORECASE,
    )

    # Format result: ('text', score)\t<path>
    p_tuple_then_path = re.compile(
        r"result:\s*(?P<tpl>\(.*\))\t(?P<path>[^\t]+\.(?:png|jpg|jpeg|bmp|webp|gif))$",
        re.IGNORECASE,
    )

    with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            path = None
            tpl = None

            m0 = p_predicts_of.search(s)
            if m0:
                path = m0.group('path').strip()
                tpl = m0.group('tpl').strip()
            else:
                m1 = p_path_then_tuple.search(s)
                if m1:
                    path = m1.group('path').strip()
                    tpl = m1.group('tpl').strip()
                else:
                    m2 = p_tuple_then_path.search(s)
                    if m2:
                        path = m2.group('path').strip()
                        tpl = m2.group('tpl').strip()

            if path is None or tpl is None:
                continue

            try:
                val = ast.literal_eval(tpl)
                pred_text = normalize_text(str(val[0] if isinstance(val, tuple) and len(val) >= 1 else val))
            except Exception:
                mtxt = re.search(r"\('(?P<txt>.*)'\s*,", tpl)
                if not mtxt:
                    continue
                pred_text = normalize_text(mtxt.group('txt'))

            pred_map[path] = pred_text

    if not pred_map:
        print(f'Warning: no predictions parsed from {raw_path}')
    else:
        print(f'Parsed predictions: {len(pred_map):,}')
    return pred_map


def evaluate_cer_ned(val_label: Path, data_root: Path, pred_raw: Path) -> Dict[str, float]:
    gt_pairs = read_gt_map(val_label)

    # Uu tien doc file ket qua tu --save_res_path neu co,
    # fallback sang log stdout khi version predict_rec.py khong ho tro tham so nay.
    pred_src = Path(pred_raw)
    if 'PRED_RES' in globals() and Path(PRED_RES).exists() and Path(PRED_RES).stat().st_size > 0:
        pred_src = Path(PRED_RES)
    print('Use prediction source:', pred_src)

    pred_map = parse_predict_output(pred_src)

    # index theo basename de xu ly truong hop path trong log khac root
    pred_by_name: Dict[str, str] = {}
    for k, v in pred_map.items():
        pred_by_name[Path(k).name] = v

    n = 0
    total_char_err = 0
    total_char_ref = 0
    total_ned = 0.0
    exact = 0

    rows = []
    for rel, gt in gt_pairs:
        img_path = str(data_root / rel)

        if img_path in pred_map:
            pred = pred_map[img_path]
        else:
            # fallback theo ten file
            pred = pred_by_name.get(Path(rel).name)
            if pred is None:
                continue

        d = levenshtein_distance(pred, gt)
        ref_len = max(1, len(gt))
        ned = d / max(len(pred), len(gt), 1)

        n += 1
        total_char_err += d
        total_char_ref += ref_len
        total_ned += ned
        exact += int(pred == gt)

        rows.append({
            'rel_path': rel,
            'gt': gt,
            'pred': pred,
            'edit_distance': d,
            'cer_sample': d / ref_len,
            'ned_sample': ned,
            'exact': int(pred == gt),
        })

    if n == 0:
        raise RuntimeError('No matched predictions. Check infer output parser/path settings.')

    metrics = {
        'num_eval_samples': n,
        'cer': total_char_err / max(total_char_ref, 1),
        'ned': total_ned / n,
        'exact_match_acc': exact / n,
    }

    df = pd.DataFrame(rows)
    display(df.head(20))
    return metrics

# Uncomment để tính metric sau infer:
metrics = evaluate_cer_ned(VAL_LABEL, WORK_DATA, PRED_RAW)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
with open(EVAL_JSON, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print('Saved metrics ->', EVAL_JSON)

# %% [markdown]
# ### Bước 13 - Quick Run Order
# Cell này chỉ in thứ tự thực thi gợi ý để tránh sót bước khi chạy trên Kaggle.

# %%
# =========================
# 13) Quick run order (manual)
# =========================
print('Suggested execution order:')
print('0.1) Install base dependencies')
print('0.2) Clone PaddleOCR + requirements')
print('0.3) Login HuggingFace via Kaggle Secrets')
print('1) Run environment setup')
print('2) (Optional) Download Manga109-s from HF')
print('3) (Optional) Extract dataset zip')
print('4) Resolve DATA_ROOT + sanity-check labels')
print('5) Define bubble->line preprocess functions')
print('6) (Optional) Preview one bubble with show_preview(...)')
print('7) Build 45k dataset with bubble->line preprocess')
print('8) Download pretrained SVTR Large (Chinese)')
print('9) Build YAML config')
print('10) Run training: run_train_with_progress(...)')
print('11) Run inference on val')
print('12) Compute CER + NED')
print('14) (Optional) Resume training to target total epoch')
print('')
print('Kaggle H100 quick checks before full train:')
print('- Keep default BATCH_PER_CARD=384 for SVTR large; fallback 320 -> 256 if OOM')
print('- Keep NUM_WORKERS around 4 (increase only if input pipeline becomes bottleneck)')
print('- Ensure train/infer share REC_IMAGE_SHAPE_STR and dict_japanese.txt path')

# %% [markdown]
# ### Bước 14 - Resume Training (Best Practice)
# Cell này dùng để train tiếp từ checkpoint gần nhất.
# 
# Quy ước quan trọng:
# - `Global.epoch_num` trong PaddleOCR là **tổng số epoch mục tiêu** (absolute target), không phải số epoch cộng thêm.
# - Ví dụ đã train tới epoch 80:
#   - đặt `epoch_num=100` => train thêm ~20 epoch (đến mốc 100).
#   - đặt `epoch_num=180` => train thêm ~100 epoch.

# %%
# =========================
# 14) Resume train from checkpoint
# =========================
import re
import yaml
from pathlib import Path


def infer_last_epoch_from_log(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    # Bat epoch tu pattern [x/y]
    matches = re.findall(r'\[(\d+)\s*/\s*\d+\]', text)
    if matches:
        return max(int(x) for x in matches)
    return 0


def prepare_resume_config(
    base_config_path: Path,
    resume_config_path: Path,
    output_dir: Path,
    target_total_epoch: int,
    train_log_path: Path,
) -> int:
    """
    target_total_epoch la MOC EPOCH TONG muon dat (khong phai cong them).
    """
    with open(base_config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ckpt_base = output_dir / 'output' / 'latest'
    ckpt_pdparams = ckpt_base.with_suffix('.pdparams')

    if not ckpt_pdparams.exists():
        raise FileNotFoundError(
            f'Checkpoint latest not found: {ckpt_pdparams}. '
            'Hay train it nhat 1 lan truoc khi resume.'
        )

    last_epoch = infer_last_epoch_from_log(train_log_path)

    if target_total_epoch <= last_epoch:
        raise ValueError(
            f'target_total_epoch={target_total_epoch} must be > last_epoch={last_epoch}. '
            'Neu muon train them 20 epoch tu moc 80, hay dat target_total_epoch=100.'
        )

    cfg['Global']['checkpoints'] = str(ckpt_base)
    cfg['Global']['pretrained_model'] = ''
    cfg['Global']['epoch_num'] = int(target_total_epoch)

    with open(resume_config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    remaining = target_total_epoch - last_epoch
    print('Resume config saved:', resume_config_path)
    print('Resume from checkpoint:', ckpt_base)
    print('Last epoch detected    :', last_epoch)
    print('Target total epoch     :', target_total_epoch)
    print('Remaining epochs       :', remaining)
    return remaining


# ===== Usage =====
# Vi du: da train den ~80, muon train tiep den 100 thi dat 100
TARGET_TOTAL_EPOCH = 80
RESUME_CONFIG_PATH = OUTPUT_ROOT / 'svtr_45k_bubbleline_resume.yml'

remaining_epochs = prepare_resume_config(
    base_config_path=CONFIG_PATH,
    resume_config_path=RESUME_CONFIG_PATH,
    output_dir=OUTPUT_ROOT,
    target_total_epoch=TARGET_TOTAL_EPOCH,
    train_log_path=TRAIN_LOG,
)

# Uncomment de chay resume:
rc = run_train_with_progress(RESUME_CONFIG_PATH, TRAIN_LOG, PADDLEOCR_DIR)
if rc != 0:
    raise RuntimeError('Resume training failed')

# %%
!i=30; rm exp_svtr_45k_bubbleline/output/iter_epoch_${i}.{pdopt,states,pdparams}

# %% [markdown]
# ### Bước 15 - Đóng gói artifact để download

# %%
### Bước 15 - Pack artifacts before stopping Kaggle
from pathlib import Path
import tarfile
from datetime import datetime

EXP_ROOT = Path('/kaggle/working/exp_svtr_45k_bubbleline')
PACK_DIR = Path('/kaggle/working')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
ARCHIVE_PATH = PACK_DIR / f'svtr_45k_artifacts_{timestamp}.tar.gz'

# Chọn các file/thư mục quan trọng
targets = [
    EXP_ROOT / 'svtr_45k_bubbleline.yml',
    EXP_ROOT / 'train.log',
    EXP_ROOT / 'eval_cer_ned.json',
    EXP_ROOT / 'val_predict_raw.txt',
    EXP_ROOT / 'val_predict_results.txt',
    EXP_ROOT / 'output' / 'best_accuracy.pdparams',
    EXP_ROOT / 'output' / 'best_accuracy.pdopt',
    EXP_ROOT / 'output' / 'best_accuracy.states',
    EXP_ROOT / 'output' / 'config.yml',
    EXP_ROOT / 'output' / 'inference',          # inference.json/.pdiparams/.yml
    EXP_ROOT / 'data' / 'dict_japanese.txt',
]

existing = [p for p in targets if p.exists()]
missing = [p for p in targets if not p.exists()]

print('Will pack:')
for p in existing:
    print(' -', p)

if missing:
    print('\nMissing (skip):')
    for p in missing:
        print(' -', p)

with tarfile.open(ARCHIVE_PATH, 'w:gz') as tar:
    for p in existing:
        tar.add(p, arcname=p.relative_to('/kaggle/working'))

print('\nArchive ready:', ARCHIVE_PATH)
print('Size (MB):', round(ARCHIVE_PATH.stat().st_size / (1024**2), 2))

# %% [markdown]
# ### Bước 16 - Load inference model + test ảnh nhanh
# 

# %%
### Bước 16 - Quick inference test on sample images
from pathlib import Path
import subprocess
import shutil

PADDLEOCR_DIR = Path('/kaggle/working/PaddleOCR')
EXP_ROOT = Path('/kaggle/working/exp_svtr_45k_bubbleline')
INFER_DIR = EXP_ROOT / 'output' / 'inference'
DICT_PATH = EXP_ROOT / 'data' / 'dict_japanese.txt'

# Tạo thư mục test input nếu chưa có
TEST_INPUT_DIR = Path('/kaggle/working/test_infer_images')
TEST_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nếu chưa có ảnh test, copy tạm 5 ảnh từ val
if not any(TEST_INPUT_DIR.glob('*')):
    val_dir = EXP_ROOT / 'data' / 'val'
    if val_dir.exists():
        for i, p in enumerate(sorted(val_dir.glob('*.png'))[:5], start=1):
            shutil.copy2(p, TEST_INPUT_DIR / p.name)
    print(f'Prepared test images in: {TEST_INPUT_DIR}')

# Verify inference artifacts
required = [INFER_DIR / 'inference.pdiparams']
if not all(p.exists() for p in required):
    raise FileNotFoundError(
        f'Inference artifacts not found in {INFER_DIR}. '
        'Please run export step first.'
    )

PRED_TEST_LOG = EXP_ROOT / 'test_predict_raw.txt'
infer_script = PADDLEOCR_DIR / 'tools' / 'infer' / 'predict_rec.py'

cmd = [
    'python', str(infer_script),
    '--rec_model_dir', str(INFER_DIR),
    '--image_dir', str(TEST_INPUT_DIR),
    '--rec_algorithm', 'SVTR',
    '--rec_image_shape', REC_IMAGE_SHAPE_STR,
    '--rec_char_dict_path', str(DICT_PATH),
    '--use_space_char', 'False',
]

with open(PRED_TEST_LOG, 'w', encoding='utf-8') as fw:
    subprocess.run(cmd, cwd=str(PADDLEOCR_DIR), stdout=fw, stderr=subprocess.STDOUT, check=True)

print('Inference log saved:', PRED_TEST_LOG)

# In nhanh vài dòng kết quả
lines = PRED_TEST_LOG.read_text(encoding='utf-8', errors='ignore').splitlines()
print('\n=== Preview predictions ===')
for ln in lines[-20:]:
    print(ln)

# %% [markdown]
# ### Bước 16b - Trực quan ảnh test và dự đoán
# Cell này đọc log inference của Bước 16, parse dự đoán rồi hiển thị ảnh kèm `pred` và `score` để kiểm tra chất lượng nhanh.

# %%
# Fix font matplotlib for Japanese text (Kaggle)
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os, subprocess, sys, warnings

# Cài font CJK nếu chưa có
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if not os.path.exists(font_path):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "japanize-matplotlib"])
    # fallback dùng japanize nếu hệ thống không có Noto CJK
    import japanize_matplotlib  # noqa: F401

# Ưu tiên Noto CJK nếu có
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"

# Tránh lỗi dấu âm unicode
plt.rcParams["axes.unicode_minus"] = False

# (tuỳ chọn) ẩn warning glyph cho sạch output
warnings.filterwarnings("ignore", message="Glyph .* missing from font")
print("Matplotlib font ready:", plt.rcParams["font.family"])

# %%
# =========================
# 16b) Visualize inference predictions
# =========================
from pathlib import Path
import re
import ast
import cv2
import matplotlib.pyplot as plt

EXP_ROOT = Path('/kaggle/working/exp_svtr_45k_bubbleline')
PRED_TEST_LOG = EXP_ROOT / 'test_predict_raw.txt'

if not PRED_TEST_LOG.exists():
    raise FileNotFoundError(f'Not found: {PRED_TEST_LOG}. Run Step 16 first.')

lines = PRED_TEST_LOG.read_text(encoding='utf-8', errors='ignore').splitlines()

# Format: Predicts of /path/img.png:('text', score)
p_predicts_of = re.compile(
    r"Predicts\s+of\s+(?P<path>.+?\.(?:png|jpg|jpeg|bmp|webp|gif)):(?P<tpl>\(.*\))$",
    re.IGNORECASE,
)

pred_items = []
for s in lines:
    m = p_predicts_of.search(s.strip())
    if not m:
        continue

    img_path = m.group('path').strip()
    tpl = m.group('tpl').strip()

    try:
        val = ast.literal_eval(tpl)
        pred_text = str(val[0]) if isinstance(val, tuple) and len(val) >= 1 else str(val)
        score = float(val[1]) if isinstance(val, tuple) and len(val) >= 2 else None
    except Exception:
        pred_text, score = tpl, None

    pred_items.append((img_path, pred_text, score))

print(f'Parsed {len(pred_items)} predictions')

show_n = min(8, len(pred_items))
if show_n == 0:
    print('No parsed predictions to visualize.')
else:
    ncols = 2
    nrows = (show_n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i in range(show_n):
        img_path, pred_text, score = pred_items[i]
        ax = axes[i]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            ax.set_title(f'Cannot read: {Path(img_path).name}')
            ax.axis('off')
            continue

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        score_str = f'{score:.3f}' if isinstance(score, float) else 'N/A'
        ax.set_title(f"{Path(img_path).name}\nPred: {pred_text}\nScore: {score_str}", fontsize=10)
        ax.axis('off')

    for j in range(show_n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


