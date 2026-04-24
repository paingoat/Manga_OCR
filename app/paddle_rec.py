"""Wrapper around PaddleOCR's tools/infer/predict_rec.py."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class PaddleRecConfig:
    paddleocr_dir: Path
    model_dir: Path
    dict_path: Path
    rec_algorithm: str = "CRNN"
    rec_image_shape: str = "3,32,320"
    use_space_char: bool = False
    use_gpu: bool = False
    enable_mkldnn: bool | None = None


def _bool_arg(value: bool) -> str:
    return "True" if value else "False"


def build_predict_command(
    cfg: PaddleRecConfig,
    image_dir: Path,
    save_res_path: Path | None = None,
) -> List[str]:
    infer_script = cfg.paddleocr_dir / "tools" / "infer" / "predict_rec.py"
    if not infer_script.exists():
        raise FileNotFoundError(
            "Cannot find PaddleOCR predict_rec.py. "
            f"Expected: {infer_script}. Pass --paddleocr-dir to the cloned PaddleOCR root."
        )
    if not cfg.model_dir.exists():
        raise FileNotFoundError(f"Recognition model directory does not exist: {cfg.model_dir}")
    if not cfg.dict_path.exists():
        raise FileNotFoundError(f"Character dictionary does not exist: {cfg.dict_path}")

    cmd = [
        sys.executable,
        str(infer_script),
        "--rec_model_dir",
        str(cfg.model_dir),
        "--image_dir",
        str(image_dir),
        "--rec_algorithm",
        cfg.rec_algorithm,
        "--rec_image_shape",
        cfg.rec_image_shape,
        "--rec_char_dict_path",
        str(cfg.dict_path),
        "--use_space_char",
        _bool_arg(cfg.use_space_char),
        "--use_gpu",
        _bool_arg(cfg.use_gpu),
    ]
    if cfg.enable_mkldnn is not None:
        cmd.extend(["--enable_mkldnn", _bool_arg(cfg.enable_mkldnn)])
    if save_res_path is not None:
        cmd.extend(["--save_res_path", str(save_res_path)])
    return cmd


def run_predict_rec_on_dir(
    cfg: PaddleRecConfig,
    image_dir: Path,
    out_log_path: Path,
    out_res_path: Path,
) -> None:
    out_log_path.parent.mkdir(parents=True, exist_ok=True)
    out_res_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_with_save = build_predict_command(cfg, image_dir, out_res_path)
    with open(out_log_path, "w", encoding="utf-8") as fw:
        rc = subprocess.run(
            cmd_with_save,
            cwd=str(cfg.paddleocr_dir),
            stdout=fw,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if rc == 0:
        return

    with open(out_log_path, "r", encoding="utf-8", errors="ignore") as fr:
        log_text = fr.read()

    if "unrecognized arguments: --save_res_path" in log_text or "error: unrecognized arguments" in log_text:
        base_cmd = build_predict_command(cfg, image_dir, None)
        with open(out_log_path, "w", encoding="utf-8") as fw:
            rc2 = subprocess.run(
                base_cmd,
                cwd=str(cfg.paddleocr_dir),
                stdout=fw,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        if rc2 == 0:
            return

    tail = "\n".join(log_text.splitlines()[-80:])
    raise RuntimeError(f"PaddleOCR inference failed. See log: {out_log_path}\n{tail}")

