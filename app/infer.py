"""End-to-end manga bubble OCR inference entrypoint."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from app.paddle_rec import PaddleRecConfig, run_predict_rec_on_dir
from app.postprocess import load_predictions
from app.preprocess import BubbleLineConfig, preprocess_file
from app.utils import (
    ensure_dir,
    list_images,
    make_timestamp,
    repo_root,
    resolve_path,
    safe_stem,
    write_json,
)


def load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configs. Install requirements.txt first.") from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section must be a mapping: {name}")
    return value


def model_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    model = _section(config, "model")
    return {
        "name": str(model.get("name", "unknown")),
        "display_name": str(model.get("display_name", model.get("name", "unknown"))),
    }


def build_preprocess_config(config: Dict[str, Any]) -> BubbleLineConfig:
    params = _section(config, "preprocess")
    valid_keys = BubbleLineConfig.__dataclass_fields__.keys()
    return BubbleLineConfig(**{k: v for k, v in params.items() if k in valid_keys})


def build_paddle_config(config: Dict[str, Any], args: argparse.Namespace, base_dir: Path) -> PaddleRecConfig:
    paths = _section(config, "paths")
    paddle = _section(config, "paddleocr")

    paddleocr_dir = resolve_path(args.paddleocr_dir or paths.get("paddleocr_dir", "PaddleOCR"), base_dir)
    model_dir = resolve_path(args.model_dir or paths.get("model_dir", "models/crnn_mobile_line/inference"), base_dir)
    dict_path = resolve_path(args.dict_path or paths.get("dict_path", "models/crnn_mobile_line/dict_japanese.txt"), base_dir)
    use_gpu = bool(paddle.get("use_gpu", False))
    if args.use_gpu:
        use_gpu = True
    if args.no_use_gpu:
        use_gpu = False

    return PaddleRecConfig(
        paddleocr_dir=paddleocr_dir,
        model_dir=model_dir,
        dict_path=dict_path,
        rec_algorithm=str(paddle.get("rec_algorithm", "CRNN")),
        rec_image_shape=str(paddle.get("rec_image_shape", "3,32,320")),
        use_space_char=bool(paddle.get("use_space_char", False)),
        use_gpu=use_gpu,
        enable_mkldnn=paddle.get("enable_mkldnn"),
    )


def preprocess_images(
    image_paths: List[Path],
    input_dir: Path,
    line_dir: Path,
    failed_dir: Path,
    cfg: BubbleLineConfig,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, src_path in enumerate(image_paths, start=1):
        rel_parent = src_path.parent.relative_to(input_dir) if src_path.parent != input_dir else Path()
        name = f"{idx:06d}_{safe_stem(src_path)}.png"
        dst_path = line_dir / rel_parent / name
        try:
            record = preprocess_file(src_path, dst_path, cfg)
            record["source_name"] = src_path.name
            record["line_name"] = dst_path.name
            records.append(record)
        except Exception as exc:
            failed_path = failed_dir / f"{idx:06d}_{src_path.name}"
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src_path, failed_path)
            except Exception:
                failed_path = None
            errors.append(
                {
                    "source": str(src_path),
                    "failed_copy": str(failed_path) if failed_path else None,
                    "error": str(exc),
                }
            )

    return records, errors


def attach_sources(
    predictions: List[Dict[str, Any]], preprocess_records: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    line_map = {Path(str(r["line_image"])).name: r for r in preprocess_records}
    output: List[Dict[str, Any]] = []
    for pred in predictions:
        key = Path(str(pred.get("image_path", ""))).name
        source = line_map.get(key)
        item = dict(pred)
        if source is not None:
            item["source_image"] = source["source"]
            item["source_name"] = source["source_name"]
            item["line_image"] = source["line_image"]
        output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manga bubble OCR inference.")
    parser.add_argument("--config", default="configs/infer.default.yaml", help="Path to YAML config.")
    parser.add_argument("--input", dest="input_dir", help="Directory containing cropped bubble images.")
    parser.add_argument("--output-root", help="Root directory for timestamped outputs.")
    parser.add_argument("--paddleocr-dir", help="Path to cloned PaddleOCR root.")
    parser.add_argument("--model-dir", help="Path to exported recognition inference model directory.")
    parser.add_argument("--dict-path", help="Path to PaddleOCR character dictionary.")
    parser.add_argument("--run-id", help="Optional run folder name. Defaults to current timestamp.")
    parser.add_argument("--use-gpu", action="store_true", help="Force PaddleOCR GPU inference.")
    parser.add_argument("--no-use-gpu", action="store_true", help="Force PaddleOCR CPU inference.")
    return parser.parse_args()


def main() -> None:
    started = time.time()
    base_dir = repo_root()
    args = parse_args()
    config_path = resolve_path(args.config, base_dir)
    config = load_config(config_path)
    paths = _section(config, "paths")
    model_info = model_metadata(config)

    input_dir = resolve_path(args.input_dir or paths.get("input_dir", "input/bubble"), base_dir)
    output_root = resolve_path(args.output_root or paths.get("output_root", "output"), base_dir)
    image_paths = list_images(input_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in input directory: {input_dir}")

    run_id = args.run_id or make_timestamp()
    run_dir = ensure_dir(output_root / run_id)
    line_dir = ensure_dir(run_dir / "line_tmp")
    failed_dir = ensure_dir(run_dir / "failed")
    raw_log_path = run_dir / "pred_raw.txt"
    result_path = run_dir / "pred_results.txt"
    predictions_path = run_dir / "predictions.json"
    manifest_path = run_dir / "manifest.json"

    preprocess_cfg = build_preprocess_config(config)
    paddle_cfg = build_paddle_config(config, args, base_dir)
    t_preprocess = time.time()
    preprocess_records, errors = preprocess_images(
        image_paths=image_paths,
        input_dir=input_dir,
        line_dir=line_dir,
        failed_dir=failed_dir,
        cfg=preprocess_cfg,
    )
    preprocess_elapsed_sec = round(time.time() - t_preprocess, 3)

    if not preprocess_records:
        manifest = {
            "run_id": run_id,
            "input_dir": str(input_dir),
            "output_dir": str(run_dir),
            "num_input_images": len(image_paths),
            "num_preprocessed": 0,
            "num_failed": len(errors),
            "errors": errors,
            "preprocess_elapsed_sec": preprocess_elapsed_sec,
            "inference_elapsed_sec": None,
            "elapsed_sec": round(time.time() - started, 3),
        }
        write_json(manifest_path, manifest)
        raise RuntimeError(f"All images failed preprocessing. See manifest: {manifest_path}")

    t_infer = time.time()
    run_predict_rec_on_dir(paddle_cfg, line_dir, raw_log_path, result_path)
    inference_elapsed_sec = round(time.time() - t_infer, 3)
    predictions = attach_sources(load_predictions(raw_log_path, result_path), preprocess_records)
    if not predictions:
        raise RuntimeError(f"No predictions parsed from PaddleOCR output. See log: {raw_log_path}")

    total_elapsed_sec = round(time.time() - started, 3)
    output_payload = {
        "run_id": run_id,
        "model": model_info,
        "input_dir": str(input_dir),
        "output_dir": str(run_dir),
        "preprocess_elapsed_sec": preprocess_elapsed_sec,
        "inference_elapsed_sec": inference_elapsed_sec,
        "elapsed_sec": total_elapsed_sec,
        "predictions": predictions,
    }
    write_json(predictions_path, output_payload)

    manifest = {
        "run_id": run_id,
        "model": model_info,
        "config_path": str(config_path),
        "input_dir": str(input_dir),
        "output_dir": str(run_dir),
        "line_tmp_dir": str(line_dir),
        "paddleocr_dir": str(paddle_cfg.paddleocr_dir),
        "model_dir": str(paddle_cfg.model_dir),
        "dict_path": str(paddle_cfg.dict_path),
        "rec_algorithm": paddle_cfg.rec_algorithm,
        "rec_image_shape": paddle_cfg.rec_image_shape,
        "enable_mkldnn": paddle_cfg.enable_mkldnn,
        "raw_log_path": str(raw_log_path),
        "result_path": str(result_path),
        "predictions_path": str(predictions_path),
        "num_input_images": len(image_paths),
        "num_preprocessed": len(preprocess_records),
        "num_predictions": len(predictions),
        "num_failed": len(errors),
        "errors": errors,
        "preprocess_elapsed_sec": preprocess_elapsed_sec,
        "inference_elapsed_sec": inference_elapsed_sec,
        "elapsed_sec": total_elapsed_sec,
    }
    write_json(manifest_path, manifest)
    print(f"Inference run complete: {run_dir}")
    print(f"Predictions: {predictions_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

