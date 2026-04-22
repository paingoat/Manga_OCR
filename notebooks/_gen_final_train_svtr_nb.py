"""One-off script to generate final_train_svtr.ipynb — run: python notebooks/_gen_final_train_svtr_nb.py"""
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
        r"""# `final_train_svtr.ipynb` — Huấn luyện SVTR Tiny (PaddleOCR) cho bài toán Rec tiếng Nhật

Notebook này chạy trên **Runpod** (RTX 3090 24GB VRAM, 32 vCPU, 125GB RAM):
- Workspace mặc định: **`/workspace`**
- Dữ liệu line đã được chuẩn bị sẵn ở bước trước, chỉ tải về và unpack
- Pretrained: **`rec_svtr_tiny_none_ctc_ch_train`** (Chinese SVTR Tiny, CTC) làm warm-start
- Training với **early stopping** (patience = `16 × eval_step`)
- Inference + đo **EM / CER / NED** trên tập **test**
- Đóng gói artifact `.tar.gz` để tải về

> Notebook chỉ tập trung vào quá trình train/infer/eval — không build dataset."""
    )
)

cells.append(
    md(
        """### Bước 1 — Gỡ stack `torch` và cài `paddlepaddle-gpu`

Runpod có thể đi kèm sẵn các thư viện torch/triton xung đột với Paddle. Gỡ trước rồi cài `paddlepaddle-gpu==3.0.0` (wheel CUDA 12.6)."""
    )
)

cells.append(
    code(
        r"""import sys
import subprocess

subprocess.run(
    [
        sys.executable, "-m", "pip", "uninstall", "-y",
        "torch", "torchvision", "torchaudio", "triton",
        "fastai", "fastcore", "fastdownload", "nbdev",
    ],
    check=False,
)

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"]
)
subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "paddlepaddle-gpu==3.0.0",
        "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    ]
)

import paddle
print("Paddle version :", paddle.__version__)
print("CUDA available :", paddle.device.is_compiled_with_cuda())
print("GPU count      :", paddle.device.cuda.device_count())
for i in range(paddle.device.cuda.device_count()):
    print(f"GPU {i}:", paddle.device.cuda.get_device_name(i))"""
    )
)

cells.append(
    md(
        """### Bước 2 — Clone PaddleOCR + cài `requirements.txt`

Clone về `/workspace/PaddleOCR`. Bỏ qua nếu thư mục đã tồn tại."""
    )
)

cells.append(
    code(
        r"""import os
import sys
import subprocess
from pathlib import Path

PADDLEOCR_DIR_BOOTSTRAP = Path("/workspace/PaddleOCR")
if not PADDLEOCR_DIR_BOOTSTRAP.is_dir():
    !git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git {PADDLEOCR_DIR_BOOTSTRAP}

# Use sys.executable -m pip to install into the SAME Python env as the kernel
# (otherwise `pip` on PATH may belong to a different environment and PaddleOCR
# deps like scikit-image / imgaug / lmdb / rapidfuzz will be missing at runtime).
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "-r", str(PADDLEOCR_DIR_BOOTSTRAP / "requirements.txt"),
])

# Safety net: explicitly ensure packages that PaddleOCR imports at startup but
# are sometimes missing from its requirements on certain environments.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "scikit-image", "imgaug", "lmdb", "rapidfuzz", "Polygon3",
    "pyclipper", "shapely", "opencv-contrib-python",
])

# Verify key imports resolve in this kernel.
import importlib
for _mod in ("skimage", "imgaug", "lmdb", "rapidfuzz", "pyclipper", "shapely"):
    try:
        importlib.import_module(_mod)
        print(f"  import {_mod:<10s} OK")
    except Exception as _e:
        print(f"  import {_mod:<10s} FAILED: {_e}")

print("PaddleOCR ready:", PADDLEOCR_DIR_BOOTSTRAP)"""
    )
)

cells.append(
    md(
        """### Bước 3 — Environment setup

Khai báo seed, đường dẫn workspace `/workspace`, vị trí dataset đã chuẩn bị (`/workspace/line_dataset`), thư mục output thí nghiệm `/workspace/exp_svtr_tiny_line` và `REC_IMAGE_SHAPE` cho SVTR."""
    )
)

cells.append(
    code(
        r"""import os
import re
import sys
import json
import math
import shutil
import random
import unicodedata
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

WORK_DIR = Path("/workspace")
PADDLEOCR_DIR = WORK_DIR / "PaddleOCR"

DATA_ROOT = WORK_DIR / "line_dataset"
TRAIN_LABEL = DATA_ROOT / "rec_gt_train.txt"
VAL_LABEL = DATA_ROOT / "rec_gt_val.txt"
TEST_LABEL = DATA_ROOT / "rec_gt_test.txt"
DICT_PATH = DATA_ROOT / "dict_japanese.txt"

OUTPUT_ROOT = WORK_DIR / "exp_svtr_tiny_line"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = OUTPUT_ROOT / "svtr_tiny_line.yml"
TRAIN_LOG = OUTPUT_ROOT / "train.log"
EVAL_JSON = OUTPUT_ROOT / "eval_test_em_cer_ned.json"

REC_IMAGE_SHAPE_STR = "3,32,320"
REC_IMAGE_SHAPE = [int(x) for x in REC_IMAGE_SHAPE_STR.split(",")]

print("WORK_DIR      :", WORK_DIR)
print("PADDLEOCR_DIR :", PADDLEOCR_DIR)
print("DATA_ROOT     :", DATA_ROOT)
print("OUTPUT_ROOT   :", OUTPUT_ROOT)
print("REC_IMAGE_SHAPE:", REC_IMAGE_SHAPE)"""
    )
)

cells.append(
    md(
        """### Bước 4 — Tải dataset từ Google Drive và giải nén

Tải file `line_dataset_60k_*.tar.gz` từ Drive bằng `gdown`, giải nén vào `/workspace/`. Sau khi unpack, cấu trúc:

```
/workspace/line_dataset/
    train/  val/  test/
    dict_japanese.txt
    rec_gt_train.txt  rec_gt_val.txt  rec_gt_test.txt
```

Idempotent: bỏ qua download/extract nếu đã có sẵn."""
    )
)

cells.append(
    code(
        r"""import sys
import subprocess
import tarfile

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])

DRIVE_FILE_ID = "1JDhDDxcZwYS8KMKDjt1k62EsGFANfxSg"
TAR_PATH = WORK_DIR / "line_dataset_60k.tar.gz"

if not DATA_ROOT.exists() or not TRAIN_LABEL.exists():
    if not TAR_PATH.exists():
        import gdown
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, str(TAR_PATH), quiet=False)

    print("Extracting:", TAR_PATH)
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(WORK_DIR)
else:
    print("DATA_ROOT already exists, skip download/extract.")

assert DATA_ROOT.exists(), f"Missing DATA_ROOT after extract: {DATA_ROOT}"
for p in [TRAIN_LABEL, VAL_LABEL, TEST_LABEL, DICT_PATH]:
    assert p.exists(), f"Missing required file: {p}"

def _count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

print("train samples:", _count_lines(TRAIN_LABEL))
print("val   samples:", _count_lines(VAL_LABEL))
print("test  samples:", _count_lines(TEST_LABEL))
print("dict  size   :", _count_lines(DICT_PATH))

print("\nSample label lines:")
with open(TRAIN_LABEL, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        print(line.rstrip("\n"))
        if i >= 4:
            break"""
    )
)

cells.append(
    md(
        """### Bước 5 — Tải pretrained `rec_svtr_tiny_none_ctc_ch_train`

Dùng SVTR Tiny tiếng Trung làm warm-start cho fine-tune tiếng Nhật. File tải về dạng `.tar`, giải nén vào `OUTPUT_ROOT/pretrained/`."""
    )
)

cells.append(
    code(
        r"""PRETRAINED_DIR = OUTPUT_ROOT / "pretrained"
PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED_URL = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_ch_train.tar"
PRETRAINED_TAR = PRETRAINED_DIR / "rec_svtr_tiny_none_ctc_ch_train.tar"
PRETRAINED_ROOT = PRETRAINED_DIR / "rec_svtr_tiny_none_ctc_ch_train"
PRETRAINED_PATH = PRETRAINED_ROOT / "best_accuracy"

if not (PRETRAINED_ROOT.exists() and any(PRETRAINED_ROOT.iterdir())):
    if not PRETRAINED_TAR.exists():
        !wget -q --show-progress -O {PRETRAINED_TAR} {PRETRAINED_URL}
    !tar -xf {PRETRAINED_TAR} -C {PRETRAINED_DIR}
else:
    print("Pretrained already extracted, skip.")

print("Pretrained URL :", PRETRAINED_URL)
print("Pretrained root:", PRETRAINED_ROOT)
print("Pretrained path:", PRETRAINED_PATH)
print("Files in root  :", sorted([p.name for p in PRETRAINED_ROOT.glob("*")]))"""
    )
)

cells.append(
    md(
        """### Bước 6 — Sinh YAML config huấn luyện SVTR Tiny

Profile cho RTX 3090 24GB:
- `batch_size_per_card = 256` (sẽ tinh chỉnh sau)
- `num_workers = 12` (32 vCPU)
- Adam + Cosine LR (`4e-4`, `warmup_epoch=2`), L2 reg `1e-5`
- `eval_step = max(200, steps_per_epoch)` (eval mỗi ~1 epoch)
- `save_epoch_step = 5` (snapshot định kỳ mỗi 5 epoch; `best_accuracy` vẫn update mỗi lần eval cải thiện)
- `Metric.main_indicator = norm_edit_dis` (≡ `1 − NED`, sát với metric đánh giá cuối)
- Architecture được lấy trực tiếp từ YAML đi kèm pretrained để khớp shape."""
    )
)

cells.append(
    code(
        r"""def count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


N_GPU = 1
BATCH_PER_CARD = int(globals().get("BATCH_PER_CARD_OVERRIDE", 256))
NUM_WORKERS = int(globals().get("NUM_WORKERS_OVERRIDE", 12))
EVAL_BATCH_PER_CARD = int(globals().get("EVAL_BATCH_PER_CARD_OVERRIDE", BATCH_PER_CARD))
EPOCH_NUM = int(globals().get("EPOCH_NUM_OVERRIDE", 100))

n_train = count_lines(TRAIN_LABEL)
steps_per_epoch = max(1, n_train // (BATCH_PER_CARD * N_GPU))
EVAL_STEP = max(200, steps_per_epoch)

pretrained_params_file = PRETRAINED_PATH.with_suffix(".pdparams")
pretrained_exists = pretrained_params_file.exists()
pretrained_for_train = str(PRETRAINED_PATH) if pretrained_exists else ""
if pretrained_exists:
    print("Using pretrained checkpoint:", PRETRAINED_PATH)
else:
    print("Pretrained .pdparams not found -> training from scratch")
    print("Missing:", pretrained_params_file)


def _load_pretrained_svtr_architecture(pretrained_root: Path) -> Dict:
    for yml_path in sorted(pretrained_root.glob("*.yml")):
        try:
            with open(yml_path, "r", encoding="utf-8") as f:
                yml_cfg = yaml.safe_load(f)
            arch = yml_cfg.get("Architecture") if isinstance(yml_cfg, dict) else None
            if isinstance(arch, dict) and arch.get("algorithm") == "SVTR":
                print("Use SVTR architecture from pretrained yaml:", yml_path.name)
                return arch
        except Exception as exc:
            print(f"Skip invalid yaml {yml_path.name}: {exc}")

    print("Fallback to built-in SVTR architecture defaults")
    return {
        "model_type": "rec",
        "algorithm": "SVTR",
        "Transform": None,
        "Backbone": {
            "name": "SVTRNet",
            "img_size": [32, 320],
            "out_channels": 192,
        },
        "Neck": {"name": "SequenceEncoder", "encoder_type": "reshape"},
        "Head": {"name": "CTCHead", "fc_decay": 1e-5},
    }


architecture_cfg = _load_pretrained_svtr_architecture(PRETRAINED_ROOT)

cfg = {
    "Global": {
        "use_gpu": True,
        "epoch_num": EPOCH_NUM,
        "log_smooth_window": 20,
        "print_batch_step": 20,
        "save_model_dir": str(OUTPUT_ROOT / "output"),
        "save_epoch_step": 5,
        "eval_batch_step": [0, EVAL_STEP],
        "cal_metric_during_train": True,
        "pretrained_model": pretrained_for_train,
        "character_dict_path": str(DICT_PATH),
        "max_text_length": 80,
        "infer_mode": False,
        "use_space_char": False,
        "save_res_path": str(OUTPUT_ROOT / "predicts.txt"),
        "seed": SEED,
    },
    "Optimizer": {
        "name": "Adam",
        "beta1": 0.9,
        "beta2": 0.999,
        "lr": {"name": "Cosine", "learning_rate": 4e-4, "warmup_epoch": 2},
        "regularizer": {"name": "L2", "factor": 1e-5},
    },
    "Architecture": architecture_cfg,
    "Loss": {"name": "CTCLoss"},
    "PostProcess": {"name": "CTCLabelDecode"},
    "Metric": {"name": "RecMetric", "main_indicator": "norm_edit_dis"},
    "Train": {
        "dataset": {
            "name": "SimpleDataSet",
            "data_dir": str(DATA_ROOT),
            "label_file_list": [str(TRAIN_LABEL)],
            "transforms": [
                {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                {"RecAug": None},
                {"CTCLabelEncode": None},
                {"RecResizeImg": {"image_shape": REC_IMAGE_SHAPE}},
                {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
            ],
        },
        "loader": {
            "shuffle": True,
            "batch_size_per_card": BATCH_PER_CARD,
            "drop_last": True,
            "num_workers": NUM_WORKERS,
            "use_shared_memory": True,
        },
    },
    "Eval": {
        "dataset": {
            "name": "SimpleDataSet",
            "data_dir": str(DATA_ROOT),
            "label_file_list": [str(VAL_LABEL)],
            "transforms": [
                {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                {"CTCLabelEncode": None},
                {"RecResizeImg": {"image_shape": REC_IMAGE_SHAPE}},
                {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
            ],
        },
        "loader": {
            "shuffle": False,
            "drop_last": False,
            "batch_size_per_card": EVAL_BATCH_PER_CARD,
            "num_workers": NUM_WORKERS,
            "use_shared_memory": True,
        },
    },
}

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print("Config saved to :", CONFIG_PATH)
print("Train samples   :", n_train)
print("Val   samples   :", count_lines(VAL_LABEL))
print("Test  samples   :", count_lines(TEST_LABEL))
print("BATCH_PER_CARD  :", BATCH_PER_CARD)
print("NUM_WORKERS     :", NUM_WORKERS)
print("EPOCH_NUM       :", EPOCH_NUM)
print("STEPS_PER_EPOCH :", steps_per_epoch)
print("EVAL_STEP       :", EVAL_STEP)
print("EARLY-STOP PATI :", 16 * EVAL_STEP, "steps (16 x eval_step)")"""
    )
)

cells.append(
    md(
        """### Bước 7 — Train với progress bar + early stopping

`run_train_with_progress` spawn `tools/train.py`, parse stdout để vẽ tqdm theo epoch và **theo dõi metric eval**:
- Mỗi lần thấy dòng eval kết thúc với `norm_edit_dis: <x>`, so với best hiện tại (đồng bộ với `main_indicator`).
- Nếu **không cải thiện trong `patience_factor=16` lần eval liên tiếp** (≈ `16 × eval_step` training steps, ~16 epoch khi eval mỗi epoch), terminate process → early stop.
- Toàn bộ stdout/stderr ghi vào `train.log` để tiện debug."""
    )
)

cells.append(
    code(
        r"""def run_train_with_progress(
    config_path: Path,
    log_path: Path,
    paddleocr_dir: Path,
    eval_step: int,
    patience_factor: int = 16,
) -> int:
    train_script = paddleocr_dir / "tools" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Cannot find train script: {train_script}")

    # Use sys.executable to guarantee the same Python interpreter (kernel)
    # that has `paddle` installed from Step 1. On Runpod, plain `python` on
    # PATH often points to a different interpreter without paddle.
    cmd = [sys.executable, str(train_script), "-c", str(config_path)]
    print("Running:", " ".join(cmd))
    print(f"Early-stopping patience: {patience_factor} evals (~{patience_factor * eval_step} steps)")

    total_epoch = None
    pbar = None
    best_metric = -1.0
    evals_since_improve = 0
    early_stopped = False

    p_epoch = re.compile(r"\[(\d+)\s*/\s*(\d+)\]")
    p_eval_metric = re.compile(
        r"(?:cur metric|best metric|metric eval).*?norm_edit_dis[:= ]\s*([0-9]*\.?[0-9]+)",
        re.IGNORECASE,
    )
    p_metric_only = re.compile(r"\bnorm_edit_dis\s*[:=]\s*([0-9]*\.?[0-9]+)")

    with open(log_path, "w", encoding="utf-8") as fw:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            cwd=str(paddleocr_dir),
        )

        try:
            for line in proc.stdout:
                fw.write(line)
                fw.flush()

                m = p_epoch.search(line)
                if m:
                    cur_ep = int(m.group(1))
                    max_ep = int(m.group(2))
                    if total_epoch is None:
                        total_epoch = max_ep
                        pbar = tqdm(total=total_epoch, desc="Training epochs", unit="ep")
                    if pbar is not None:
                        pbar.n = min(cur_ep, total_epoch)
                        pbar.refresh()

                low = line.lower()
                is_eval_line = ("cur metric" in low) or ("best metric" in low) or (
                    "metric eval" in low and "norm_edit_dis" in low
                )

                if is_eval_line:
                    am = p_eval_metric.search(line) or p_metric_only.search(line)
                    if am:
                        try:
                            cur_metric = float(am.group(1))
                        except ValueError:
                            cur_metric = None

                        if cur_metric is not None:
                            if cur_metric > best_metric + 1e-6:
                                best_metric = cur_metric
                                evals_since_improve = 0
                                print(f"[ES] new best norm_edit_dis={best_metric:.4f}")
                            else:
                                evals_since_improve += 1
                                print(
                                    f"[ES] no improve ({evals_since_improve}/{patience_factor}), "
                                    f"cur norm_edit_dis={cur_metric:.4f} best={best_metric:.4f}"
                                )
                                if evals_since_improve >= patience_factor:
                                    early_stopped = True
                                    print(
                                        f"[ES] Early stopped after {patience_factor} evaluations "
                                        f"without improvement (~{patience_factor * eval_step} steps)"
                                    )
                                    proc.terminate()
                                    break

                if "best metric" in low or low.startswith("eval ") or "metric eval" in low:
                    print(line.rstrip())

            rc = proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except Exception:
                    proc.kill()

    if pbar is not None:
        pbar.refresh()
        pbar.close()

    if early_stopped:
        print("Train finished by EARLY STOP. Best norm_edit_dis =", best_metric)
        return 0

    print("Train finished with code:", rc)
    print("Train log:", log_path)

    if rc != 0:
        print("\n=== Last 120 lines of train log ===")
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as fr:
                lines = fr.readlines()
            for line in lines[-120:]:
                print(line.rstrip("\n"))
        except Exception as e:
            print("Could not read train log tail:", e)

    return rc


rc = run_train_with_progress(
    CONFIG_PATH, TRAIN_LOG, PADDLEOCR_DIR, eval_step=EVAL_STEP, patience_factor=16
)
if rc != 0:
    raise RuntimeError("Training failed")"""
    )
)

cells.append(
    md(
        """### Bước 8 — Patch config cho inference (`SVTRRecResizeImg`)

PaddleOCR yêu cầu transform `SVTRRecResizeImg` cho SVTR khi infer. Đổi `RecResizeImg` -> `SVTRRecResizeImg` (`padding=False`) cho cả Train/Eval section."""
    )
)

cells.append(
    code(
        r"""with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg_patched = yaml.safe_load(f)


def patch_transforms(transforms):
    if not transforms:
        return
    for item in transforms:
        if not isinstance(item, dict):
            continue
        if "RecResizeImg" in item:
            old = item.pop("RecResizeImg")
            shape = old.get("image_shape")
            item["SVTRRecResizeImg"] = {"image_shape": shape, "padding": False}


for split in ("Train", "Eval"):
    ds = cfg_patched.get(split, {}).get("dataset", {})
    patch_transforms(ds.get("transforms"))

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg_patched, f, sort_keys=False, allow_unicode=True)

print("Patched SVTRRecResizeImg in:", CONFIG_PATH)"""
    )
)

cells.append(
    md(
        """### Bước 9 — Inference trên tập **test**

Export checkpoint `best_accuracy` thành inference model, sau đó chạy `predict_rec.py` trên thư mục `test/` để lấy kết quả phục vụ tính EM/CER/NED."""
    )
)

cells.append(
    code(
        r"""BEST_CKPT_PREFIX = OUTPUT_ROOT / "output" / "best_accuracy"
INFER_MODEL_DIR = OUTPUT_ROOT / "output" / "inference"
TEST_IMAGE_DIR = DATA_ROOT / "test"
PRED_RAW = OUTPUT_ROOT / "test_predict_raw.txt"
PRED_RES = OUTPUT_ROOT / "test_predict_results.txt"


def export_inference_model_if_needed(
    config_path: Path,
    paddleocr_dir: Path,
    checkpoint_prefix: Path,
    infer_dir: Path,
) -> None:
    export_script = paddleocr_dir / "tools" / "export_model.py"
    if not export_script.exists():
        raise FileNotFoundError(f"Cannot find export script: {export_script}")

    infer_json = infer_dir / "inference.json"
    infer_pdiparams = infer_dir / "inference.pdiparams"
    infer_pdmodel = infer_dir / "inference.pdmodel"

    if (infer_json.exists() and infer_pdiparams.exists()) or (
        infer_pdmodel.exists() and infer_pdiparams.exists()
    ):
        print("Inference model exists, skip export:", infer_dir)
        return

    infer_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(export_script),
        "-c", str(config_path),
        "-o",
        f"Global.pretrained_model={checkpoint_prefix}",
        f"Global.save_inference_dir={infer_dir}",
    ]
    print("Export inference model...")
    subprocess.run(cmd, cwd=str(paddleocr_dir), check=True)


def run_predict_rec_on_dir(
    paddleocr_dir: Path,
    image_dir: Path,
    infer_model_dir: Path,
    out_log_path: Path,
    out_res_path: Path,
) -> None:
    infer_script = paddleocr_dir / "tools" / "infer" / "predict_rec.py"
    if not infer_script.exists():
        raise FileNotFoundError(f"Cannot find infer script: {infer_script}")

    base_cmd = [
        sys.executable, str(infer_script),
        "--rec_model_dir", str(infer_model_dir),
        "--image_dir", str(image_dir),
        "--rec_algorithm", "SVTR",
        "--rec_image_shape", REC_IMAGE_SHAPE_STR,
        "--rec_char_dict_path", str(DICT_PATH),
        "--use_space_char", "False",
    ]

    cmd_with_save = base_cmd + ["--save_res_path", str(out_res_path)]

    with open(out_log_path, "w", encoding="utf-8") as fw:
        rc = subprocess.run(
            cmd_with_save, cwd=str(paddleocr_dir), stdout=fw, stderr=subprocess.STDOUT
        ).returncode

    if rc == 0:
        print("Inference done with --save_res_path:", out_res_path)
        return

    with open(out_log_path, "r", encoding="utf-8", errors="ignore") as fr:
        log_text = fr.read()

    if ("unrecognized arguments: --save_res_path" in log_text) or (
        "error: unrecognized arguments" in log_text
    ):
        print("predict_rec.py does not support --save_res_path, fallback to stdout log parsing")
        with open(out_log_path, "w", encoding="utf-8") as fw:
            rc2 = subprocess.run(
                base_cmd, cwd=str(paddleocr_dir), stdout=fw, stderr=subprocess.STDOUT
            ).returncode
        if rc2 == 0:
            print("Inference done (stdout mode). Log saved to:", out_log_path)
            return

    print("\n=== Inference log tail ===")
    with open(out_log_path, "r", encoding="utf-8", errors="ignore") as fr:
        lines = fr.readlines()
    for line in lines[-120:]:
        print(line.rstrip("\n"))

    raise RuntimeError(f"Inference failed. See log: {out_log_path}")


print("BEST_CKPT_PREFIX:", BEST_CKPT_PREFIX)
print("INFER_MODEL_DIR :", INFER_MODEL_DIR)
print("TEST_IMAGE_DIR  :", TEST_IMAGE_DIR)
print("PRED_RAW (LOG)  :", PRED_RAW)
print("PRED_RES (RES)  :", PRED_RES)

export_inference_model_if_needed(CONFIG_PATH, PADDLEOCR_DIR, BEST_CKPT_PREFIX, INFER_MODEL_DIR)
run_predict_rec_on_dir(PADDLEOCR_DIR, TEST_IMAGE_DIR, INFER_MODEL_DIR, PRED_RAW, PRED_RES)"""
    )
)

cells.append(
    md(
        """### Bước 10 — Tính EM / CER / NED trên tập test

- **EM**: Exact Match (chuỗi pred trùng GT sau khi normalize).
- **CER**: total Levenshtein distance / total ref char length.
- **NED**: trung bình `edit_distance / max(len(pred), len(gt))`.

Ưu tiên đọc file `--save_res_path` (ổn định), fallback parse từ stdout log."""
    )
)

cells.append(
    code(
        r"""def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


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
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = cur
    return prev[-1]


def read_gt_map(label_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            rel, txt = line.split("\t", 1)
            rows.append((rel, normalize_text(txt)))
    return rows


def parse_predict_output(raw_path: Path) -> Dict[str, str]:
    import ast

    pred_map: Dict[str, str] = {}

    p_predicts_of = re.compile(
        r"Predicts\s+of\s+(?P<path>.+?\.(?:png|jpg|jpeg|bmp|webp|gif)):(?P<tpl>\(.*\))$",
        re.IGNORECASE,
    )
    p_path_then_tuple = re.compile(
        r"(?P<path>[^\t]+\.(?:png|jpg|jpeg|bmp|webp|gif))\t(?P<tpl>\(.*\))$",
        re.IGNORECASE,
    )
    p_tuple_then_path = re.compile(
        r"result:\s*(?P<tpl>\(.*\))\t(?P<path>[^\t]+\.(?:png|jpg|jpeg|bmp|webp|gif))$",
        re.IGNORECASE,
    )

    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            path = None
            tpl = None

            m0 = p_predicts_of.search(s)
            if m0:
                path = m0.group("path").strip()
                tpl = m0.group("tpl").strip()
            else:
                m1 = p_path_then_tuple.search(s)
                if m1:
                    path = m1.group("path").strip()
                    tpl = m1.group("tpl").strip()
                else:
                    m2 = p_tuple_then_path.search(s)
                    if m2:
                        path = m2.group("path").strip()
                        tpl = m2.group("tpl").strip()

            if path is None or tpl is None:
                continue

            try:
                val = ast.literal_eval(tpl)
                pred_text = normalize_text(
                    str(val[0] if isinstance(val, tuple) and len(val) >= 1 else val)
                )
            except Exception:
                mtxt = re.search(r"\('(?P<txt>.*)'\s*,", tpl)
                if not mtxt:
                    continue
                pred_text = normalize_text(mtxt.group("txt"))

            pred_map[path] = pred_text

    if not pred_map:
        print(f"Warning: no predictions parsed from {raw_path}")
    else:
        print(f"Parsed predictions: {len(pred_map):,}")
    return pred_map


def evaluate_em_cer_ned(
    gt_label: Path, data_root: Path, pred_raw: Path, pred_res: Optional[Path] = None
) -> Dict[str, float]:
    gt_pairs = read_gt_map(gt_label)

    pred_src = Path(pred_raw)
    if pred_res is not None and Path(pred_res).exists() and Path(pred_res).stat().st_size > 0:
        pred_src = Path(pred_res)
    print("Use prediction source:", pred_src)

    pred_map = parse_predict_output(pred_src)
    pred_by_name: Dict[str, str] = {Path(k).name: v for k, v in pred_map.items()}

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
            "rel_path": rel,
            "gt": gt,
            "pred": pred,
            "edit_distance": d,
            "cer_sample": d / ref_len,
            "ned_sample": ned,
            "exact": int(pred == gt),
        })

    if n == 0:
        raise RuntimeError("No matched predictions. Check infer output parser/path settings.")

    metrics = {
        "num_eval_samples": n,
        "exact_match_acc": exact / n,
        "cer": total_char_err / max(total_char_ref, 1),
        "ned": total_ned / n,
    }

    df = pd.DataFrame(rows)
    try:
        display(df.head(20))
    except NameError:
        print(df.head(20))
    return metrics


metrics = evaluate_em_cer_ned(TEST_LABEL, DATA_ROOT, PRED_RAW, PRED_RES)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
with open(EVAL_JSON, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("Saved metrics ->", EVAL_JSON)"""
    )
)

cells.append(
    md(
        """### Bước 11 — Resume training (đang để comment)

Khi muốn train tiếp từ checkpoint `latest`, **uncomment toàn bộ cell** dưới đây và chỉnh `TARGET_TOTAL_EPOCH` (đây là **mốc tổng** muốn đạt, không phải số epoch cộng thêm).

> Early stopping vẫn áp dụng với patience = `16 × eval_step`."""
    )
)

cells.append(
    code(
        r"""# def infer_last_epoch_from_log(log_path: Path) -> int:
#     if not log_path.exists():
#         return 0
#     text = log_path.read_text(encoding="utf-8", errors="ignore")
#     matches = re.findall(r"\[(\d+)\s*/\s*\d+\]", text)
#     if matches:
#         return max(int(x) for x in matches)
#     return 0
#
#
# def prepare_resume_config(
#     base_config_path: Path,
#     resume_config_path: Path,
#     output_dir: Path,
#     target_total_epoch: int,
#     train_log_path: Path,
# ) -> int:
#     with open(base_config_path, "r", encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)
#
#     ckpt_base = output_dir / "output" / "latest"
#     ckpt_pdparams = ckpt_base.with_suffix(".pdparams")
#     if not ckpt_pdparams.exists():
#         raise FileNotFoundError(
#             f"Checkpoint latest not found: {ckpt_pdparams}. Train at least once first."
#         )
#
#     last_epoch = infer_last_epoch_from_log(train_log_path)
#     if target_total_epoch <= last_epoch:
#         raise ValueError(
#             f"target_total_epoch={target_total_epoch} must be > last_epoch={last_epoch}."
#         )
#
#     cfg["Global"]["checkpoints"] = str(ckpt_base)
#     cfg["Global"]["pretrained_model"] = ""
#     cfg["Global"]["epoch_num"] = int(target_total_epoch)
#
#     with open(resume_config_path, "w", encoding="utf-8") as f:
#         yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
#
#     remaining = target_total_epoch - last_epoch
#     print("Resume config saved:", resume_config_path)
#     print("Resume from checkpoint:", ckpt_base)
#     print("Last epoch detected   :", last_epoch)
#     print("Target total epoch    :", target_total_epoch)
#     print("Remaining epochs      :", remaining)
#     return remaining
#
#
# TARGET_TOTAL_EPOCH = 150
# RESUME_CONFIG_PATH = OUTPUT_ROOT / "svtr_tiny_line_resume.yml"
#
# remaining_epochs = prepare_resume_config(
#     base_config_path=CONFIG_PATH,
#     resume_config_path=RESUME_CONFIG_PATH,
#     output_dir=OUTPUT_ROOT,
#     target_total_epoch=TARGET_TOTAL_EPOCH,
#     train_log_path=TRAIN_LOG,
# )
#
# rc = run_train_with_progress(
#     RESUME_CONFIG_PATH, TRAIN_LOG, PADDLEOCR_DIR, eval_step=EVAL_STEP, patience_factor=16
# )
# if rc != 0:
#     raise RuntimeError("Resume training failed")"""
    )
)

cells.append(
    md(
        """### Bước 12 — Đóng gói artifact `.tar.gz` để tải về

Gom các file/thư mục quan trọng (config, log, metric, predictions, best checkpoint, inference model, dict) vào `/workspace/svtr_tiny_artifacts_{timestamp}.tar.gz`."""
    )
)

cells.append(
    code(
        r"""import tarfile
from datetime import datetime

EXP_ROOT = OUTPUT_ROOT
PACK_DIR = WORK_DIR
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ARCHIVE_PATH = PACK_DIR / f"svtr_tiny_artifacts_{timestamp}.tar.gz"

targets = [
    CONFIG_PATH,
    TRAIN_LOG,
    EVAL_JSON,
    PRED_RAW,
    PRED_RES,
    EXP_ROOT / "output" / "best_accuracy.pdparams",
    EXP_ROOT / "output" / "best_accuracy.pdopt",
    EXP_ROOT / "output" / "best_accuracy.states",
    EXP_ROOT / "output" / "config.yml",
    EXP_ROOT / "output" / "inference",
    DICT_PATH,
]

existing = [p for p in targets if p.exists()]
missing = [p for p in targets if not p.exists()]

print("Will pack:")
for p in existing:
    print(" -", p)

if missing:
    print("\nMissing (skip):")
    for p in missing:
        print(" -", p)

with tarfile.open(ARCHIVE_PATH, "w:gz") as tar:
    for p in existing:
        try:
            arcname = p.relative_to(WORK_DIR)
        except ValueError:
            arcname = p.name
        tar.add(p, arcname=str(arcname))

print("\nArchive ready:", ARCHIVE_PATH)
print("Size (MB):", round(ARCHIVE_PATH.stat().st_size / (1024 ** 2), 2))"""
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

OUT = Path(__file__).resolve().parent / "final_train_svtr.ipynb"
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", OUT, "cells:", len(cells))
