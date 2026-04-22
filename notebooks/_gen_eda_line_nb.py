"""Generate notebooks/eda_line_dataset.ipynb — run: python notebooks/_gen_eda_line_nb.py"""
import json
from pathlib import Path


def md(s: str):
    lines = s.strip().split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines]}


def code(s: str):
    lines = s.strip().split("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [ln + "\n" for ln in lines],
        "outputs": [],
        "execution_count": None,
    }


cells = []

cells.append(
    md(
        r"""# EDA — `line_dataset` (manga OCR, line recognition)

Notebook chạy trên **Kaggle** (CPU đủ). Phân tích nhanh tập **line**: nhãn, từ điển, kích thước ảnh, độ sáng.

**Cấu trúc thư mục** (sau khi Add Data):

- `train/`, `val/`, `test/` — ảnh `.png`
- `rec_gt_train.txt`, `rec_gt_val.txt`, `rec_gt_test.txt` — mỗi dòng: `đường_dẫn_ảnh\t nhãn`
- `dict_japanese.txt` — một ký tự / dòng

**Đường dẫn**: mặc định dataset Kaggle  
`/kaggle/input/datasets/narcolepsyy/jap-ocr-line-60k-v2/line_dataset`  
Nếu bạn đổi tên hoặc thêm Dataset khác, sửa biến `DATA_ROOT` trong cell import."""
    )
)

cells.append(
    md(
        """### Bước 1 — Cài thư viện

`pandas`, `matplotlib`, `seaborn`, `opencv`, `Pillow`, `tqdm`, `wordcloud`. Không Paddle."""
    )
)

cells.append(
    code(
        r"""import sys
import subprocess

# Kaggle / Debian: font CJK cho matplotlib + wordcloud (tranh tofu / DejaVu)
if sys.platform.startswith("linux"):
    try:
        subprocess.run(
            ["apt-get", "update", "-qq"],
            check=False,
            timeout=180,
            capture_output=True,
        )
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
            check=False,
            timeout=300,
            capture_output=True,
        )
    except Exception:
        pass

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"]
)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "opencv-python-headless",
        "pillow",
        "tqdm",
        "wordcloud",
    ]
)
print("OK.")"""
    )
)

cells.append(
    md(
        """### Bước 2 — Import, `DATA_ROOT`, đọc nhãn và dict

Đọc ba file `rec_gt_*.txt` thành DataFrame; đọc `dict_japanese.txt` thành tập ký tự."""
    )
)

cells.append(
    code(
        r"""from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from tqdm.auto import tqdm

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.unicode_minus"] = False


def resolve_cjk_font_path():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
    ]
    from pathlib import Path as _P

    for p in candidates:
        if _P(p).is_file():
            return p
    fonts_root = _P("/usr/share/fonts")
    if fonts_root.is_dir():
        for name in (
            "NotoSansCJK-Regular.ttc",
            "NotoSansCJK-Medium.ttc",
            "NotoSansCJKjp-Regular.otf",
        ):
            for p in fonts_root.rglob(name):
                if p.is_file():
                    return str(p)
    return None


FONT_PATH_CJK = resolve_cjk_font_path()
if FONT_PATH_CJK:
    try:
        fm.fontManager.addfont(FONT_PATH_CJK)
        _fp = fm.FontProperties(fname=FONT_PATH_CJK)
        _fname = _fp.get_name()
        plt.rcParams["font.sans-serif"] = [_fname] + [
            x for x in plt.rcParams.get("font.sans-serif", []) if x != _fname
        ]
        plt.rcParams["font.family"] = "sans-serif"
    except Exception as e:
        print("Font CJK register warning:", e)
    print("FONT_PATH_CJK =", FONT_PATH_CJK)
else:
    print("Warning: khong tim thay Noto CJK — bar/character cloud co the bao loi glyph.")

# --- Kaggle Input (mặc định: jap-ocr-line-60k-v2 / line_dataset) ---
_CANDIDATES = [
    Path("/kaggle/input/datasets/narcolepsyy/jap-ocr-line-60k-v2/line_dataset"),
    Path("/kaggle/input/jap-ocr-line-60k-v2/line_dataset"),
    Path("/kaggle/input/line_dataset"),
    Path("/kaggle/input/line_dataset/line_dataset"),
]
DATA_ROOT = _CANDIDATES[0]
for c in _CANDIDATES:
    if (c / "rec_gt_train.txt").exists():
        DATA_ROOT = c
        break
else:
    print("Warning: rec_gt_train.txt not found; tried:", [str(x) for x in _CANDIDATES])


def load_rec_gt(root: Path, split: str) -> pd.DataFrame:
    path = root / f"rec_gt_{split}.txt"
    rows: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            rel, text = line.split("\t", 1)
            rows.append((rel.strip(), text.strip()))
    return pd.DataFrame(rows, columns=["rel_path", "text"])


def load_charset_dict(path: Path) -> Set[str]:
    s: Set[str] = set()
    if not path.exists():
        return s
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if line:
                s.add(line)
    return s


df_train = load_rec_gt(DATA_ROOT, "train")
df_val = load_rec_gt(DATA_ROOT, "val")
df_test = load_rec_gt(DATA_ROOT, "test")

dict_chars = load_charset_dict(DATA_ROOT / "dict_japanese.txt")

print("DATA_ROOT:", DATA_ROOT)
print("train / val / test rows:", len(df_train), len(df_val), len(df_test))
print("|dict| =", len(dict_chars))"""
    )
)

cells.append(
    md(
        """### EDA 1 — Quy mô train / val / test

Biểu đồ cột, **biểu đồ tròn** (tỷ lệ %), và bảng số mẫu."""
    )
)

cells.append(
    code(
        r"""labels = ["train", "val", "test"]
counts = [len(df_train), len(df_val), len(df_test)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].bar(labels, counts, color=["#4C72B0", "#55A868", "#C44E52"])
axes[0].set_title("Số mẫu theo split")
axes[0].set_ylabel("Count")

axes[1].pie(
    counts,
    labels=labels,
    autopct="%1.1f%%",
    colors=["#4C72B0", "#55A868", "#C44E52"],
    startangle=90,
)
axes[1].set_title("Tỷ lệ train / val / test")

plt.tight_layout()
plt.show()

summary = pd.DataFrame({"split": labels, "n_samples": counts})
summary["pct"] = (summary["n_samples"] / summary["n_samples"].sum() * 100).round(2)
print(summary.to_string(index=False))"""
    )
)
cells.append(
    md(
        """### EDA 2 — Độ dài chuỗi nhãn (số ký tự)

Histogram theo từng split; in mean / median / p95."""
    )
)

cells.append(
    code(
        r"""fig, ax = plt.subplots(figsize=(9, 4))
for name, dfi, c in [("train", df_train, "#4C72B0"), ("val", df_val, "#55A868"), ("test", df_test, "#C44E52")]:
    lens = dfi["text"].str.len()
    ax.hist(lens, bins=40, alpha=0.45, label=name, color=c)
ax.set_xlabel("Do dai nhan (ky tu)")
ax.set_ylabel("So mau")
ax.set_title("Phan phoi do dai chuoi theo split")
ax.legend()
plt.tight_layout()
plt.show()

for name, dfi in [("train", df_train), ("val", df_val), ("test", df_test)]:
    lens = dfi["text"].str.len()
    p95 = lens.quantile(0.95)
    p99 = lens.quantile(0.99)
    print(
        name,
        "mean=%.2f median=%.0f p95=%.0f p99=%.0f max=%d"
        % (lens.mean(), lens.median(), p95, p99, lens.max()),
    )
    hi = int((lens > p99).sum())
    print(f"  samples > p99 length ({p99:.0f} chars): {hi}")
    over_80 = int((lens > 80).sum())
    print(f"  samples > 80 chars (SVTR T cap): {over_80} ({over_80 / max(len(lens), 1) * 100:.2f}%)")"""
    )
)

cells.append(
    md(
        """### EDA 3 — Tần suất ký tự (top-40)

Bar chart ký tự xuất hiện nhiều nhất (gộp train+val+test)."""
    )
)

cells.append(
    code(
        r"""all_text = "".join(df_train["text"]) + "".join(df_val["text"]) + "".join(df_test["text"])
freq = Counter(all_text)
topn = 40
chars, counts2 = zip(*freq.most_common(topn))
chars = list(chars)[::-1]
counts2 = list(counts2)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(len(chars)), counts2, color="steelblue")
ax.set_yticks(range(len(chars)))
ax.set_yticklabels(chars, fontsize=7)
ax.set_xlabel("Tan suat")
ax.set_title(f"Top {topn} ky tu (tat ca split)")
plt.tight_layout()
plt.show()

rare = sum(1 for _, v in freq.items() if v == 1)
print("Ky tu chi xuat hien 1 lan (trong toan bo nhan):", rare)"""
    )
)

cells.append(
    md(
        """### EDA 3b — Character cloud (tần suất ký tự)

**Character cloud** (không phải word cloud): mỗi **ký tự** là một token, kích thước ~ tần suất. Cần font **Noto CJK** (cell pip cài `fonts-noto-cjk` trên Linux)."""
    )
)

cells.append(
    code(
        r"""from wordcloud import WordCloud

freq_dict = dict(freq)
font_path = FONT_PATH_CJK or resolve_cjk_font_path()

if not font_path:
    raise RuntimeError(
        "Khong tim thay font Noto CJK. Chay lai cell pip (fonts-noto-cjk) hoac dat bien FONT_PATH_CJK."
    )

wc = WordCloud(
    font_path=font_path,
    width=900,
    height=450,
    background_color="white",
    max_words=500,
    relative_scaling=0.4,
    colormap="viridis",
).generate_from_frequencies(freq_dict)

fig, ax = plt.subplots(figsize=(11, 5))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
ax.set_title("Character cloud (tan suat ky tu)")
plt.tight_layout()
plt.show()"""
    )
)

cells.append(
    md(
        """### EDA 4 — Nhãn vs `dict_japanese.txt`

Ký tự có trong nhãn nhưng không có trong dict; tỷ lệ **ký tự** (occurrence) trong / ngoài dict."""
    )
)

cells.append(
    code(
        r"""label_chars = set(all_text)
missing = sorted(label_chars - dict_chars)
print("So ky tu trong nhan ma khong co trong dict:", len(missing))
if missing[:50]:
    print("Vi du (toi da 50):", "".join(missing[:50]))

in_dict_occ = 0
out_dict_occ = 0
for ch, cnt in freq.items():
    if ch in dict_chars:
        in_dict_occ += cnt
    else:
        out_dict_occ += cnt
tot = in_dict_occ + out_dict_occ
if tot > 0:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        [in_dict_occ, out_dict_occ],
        labels=["Trong dict", "Ngoai dict"],
        autopct="%1.1f%%",
        colors=["#55A868", "#DD8452"],
        startangle=90,
    )
    ax.set_title("Ty le ky tu (occurrence)")
    plt.tight_layout()
    plt.show()
    print("in_dict occ:", in_dict_occ, "out_dict occ:", out_dict_occ)"""
    )
)

cells.append(
    md(
        """### EDA 5 — Ảnh: chiều rộng, cao, aspect ratio

Đọc kích thước từng ảnh (có `tqdm`). Có thể lâu với ~60k ảnh; giới hạn `MAX_IMAGES_PER_SPLIT = None` để quét hết, hoặc số nguyên để sample."""
    )
)

cells.append(
    code(
        r"""import cv2

MAX_IMAGES_PER_SPLIT = None  # dat vi du 5000 de thu nhanh
RNG = np.random.default_rng(42)


def collect_sizes(df: pd.DataFrame, split: str, cap: int | None):
    ws, hs = [], []
    idxs = np.arange(len(df))
    if cap is not None and len(idxs) > cap:
        idxs = RNG.choice(idxs, size=cap, replace=False)
    sub = df.iloc[idxs]
    for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"size {split}"):
        p = DATA_ROOT / row["rel_path"]
        if not p.exists():
            continue
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        h, w = im.shape[:2]
        ws.append(w)
        hs.append(h)
    return ws, hs


w_tr, h_tr = collect_sizes(df_train, "train", MAX_IMAGES_PER_SPLIT)
w_va, h_va = collect_sizes(df_val, "val", MAX_IMAGES_PER_SPLIT)
w_te, h_te = collect_sizes(df_test, "test", MAX_IMAGES_PER_SPLIT)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].hist([np.array(w_tr), np.array(w_va), np.array(w_te)], bins=40, label=["train", "val", "test"], alpha=0.6)
axes[0, 0].set_title("Chieu rong (px)")
axes[0, 0].legend()
axes[0, 1].hist([np.array(h_tr), np.array(h_va), np.array(h_te)], bins=40, label=["train", "val", "test"], alpha=0.6)
axes[0, 1].set_title("Chieu cao (px)")
axes[0, 1].legend()
ar_tr = np.array(w_tr) / np.maximum(np.array(h_tr), 1)
ar_va = np.array(w_va) / np.maximum(np.array(h_va), 1)
ar_te = np.array(w_te) / np.maximum(np.array(h_te), 1)
axes[1, 0].hist([ar_tr, ar_va, ar_te], bins=40, label=["train", "val", "test"], alpha=0.6)
axes[1, 0].set_title("Aspect ratio W/H")
axes[1, 0].legend()
axes[1, 1].axis("off")
plt.tight_layout()
plt.show()"""
    )
)

cells.append(
    md(
        """### EDA 6 — Độ sáng (mean grayscale)

Trên mẫu tối đa 5000 ảnh / split (có thể chỉnh `BRIGHT_SAMPLE`)."""
    )
)

cells.append(
    code(
        r"""BRIGHT_SAMPLE = 5000


def sample_mean_gray(df: pd.DataFrame, split: str, cap: int):
    idxs = np.arange(len(df))
    if len(idxs) > cap:
        idxs = RNG.choice(idxs, size=cap, replace=False)
    means = []
    sub = df.iloc[idxs]
    for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"bright {split}"):
        p = DATA_ROOT / row["rel_path"]
        if not p.exists():
            continue
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        means.append(float(g.mean()))
    return means


m_tr = sample_mean_gray(df_train, "train", BRIGHT_SAMPLE)
m_va = sample_mean_gray(df_val, "val", BRIGHT_SAMPLE)
m_te = sample_mean_gray(df_test, "test", BRIGHT_SAMPLE)

plt.figure(figsize=(8, 4))
plt.hist([m_tr, m_va, m_te], bins=40, label=["train", "val", "test"], alpha=0.65)
plt.xlabel("Mean grayscale (0-255)")
plt.title("Do sang trung binh (mau)")
plt.legend()
plt.tight_layout()
plt.show()"""
    )
)

cells.append(
    md(
        """### Kết

Chỉnh `DATA_ROOT` nếu Input khác; chạy tuần tự. CPU đủ; không cần GPU."""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

OUT = Path(__file__).resolve().parent / "final_eda_line_dataset.ipynb"
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", OUT, "cells:", len(cells))
