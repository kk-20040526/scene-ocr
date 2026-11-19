# -*- coding: utf-8 -*-
"""
生成 predictions.jsonl（严格符合任务书 4.5 节）

用法：
  # 正式：对待预测图片目录
  .venv\\Scripts\\python.exe deploy\\generate_predictions.py --image-dir <待预测目录>
  # 自测（用 valid 图片；valid 文件名在 data/ppocr/split.json 中）
  .venv\\Scripts\\python.exe deploy\\generate_predictions.py --image-dir data/train/img --limit 50

输出默认：项目根/predictions.jsonl
每行：
  {"image_id": "xxx.jpg", "predictions": [
      {"points": [x1,y1,...,x4,y4], "transcription": "...", "confidence": 0.93}, ...]}
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_engine import PROJ, OCREngine, imread_unicode  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_finite_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and \
        math.isfinite(float(x))


def box_to_points(box, h, w):
    """四点 -> 8 个 int 坐标；裁剪到图内；返回 points 或 None（无效框）。"""
    pts = np.asarray(box, dtype=np.float64).reshape(4, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    try:
        poly = Polygon(pts).buffer(0)
        if poly.is_empty or poly.area <= 1.0 or not poly.is_valid:
            return None
    except Exception:
        return None
    coords = [int(round(v)) for v in pts.reshape(-1).tolist()]
    # 再次确保 8 个有限值
    if len(coords) != 8 or not all(math.isfinite(float(v)) for v in coords):
        return None
    return coords


def list_images(image_dir):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(image_dir, "*" + ext)))
        files.extend(glob.glob(os.path.join(image_dir, "*" + ext.upper())))
    return sorted(set(files))


def generate(image_dir, output_path, det_weights, rec_weights,
             limit=None, limit_side_len=960):
    print("loading OCR engine ...")
    engine = OCREngine(det_weights=det_weights, rec_weights=rec_weights,
                       limit_side_len=limit_side_len)
    images = list_images(image_dir)
    if limit:
        images = images[:limit]
    if not images:
        raise RuntimeError("no images found in %s" % image_dir)
    print("images to predict: %d" % len(images))

    seen_ids = set()
    fail_read = []
    rows = []
    for i, path in enumerate(images):
        image_id = os.path.basename(path)
        if image_id in seen_ids:
            raise RuntimeError("duplicate image_id: %s" % image_id)
        seen_ids.add(image_id)
        img = imread_unicode(path)
        if img is None or getattr(img, "shape", None) is None:
            fail_read.append(image_id)
            # 任务书：读取失败须排查后重跑，不得伪装为无文字 -> 不写该行
            continue
        h, w = img.shape[:2]
        boxes, texts, scores = engine(img, keep_all=True)
        preds = []
        for box, text, score in zip(boxes, texts, scores):
            pts = box_to_points(box, h, w)
            if pts is None:
                continue
            text = text if isinstance(text, str) else ""
            text = text.strip()
            if not text:
                conf = 0.0
            else:
                conf = float(score) if is_finite_num(score) else 0.0
                conf = min(1.0, max(0.0, conf))
            preds.append({
                "points": pts,
                "transcription": text,
                "confidence": round(conf, 4),
            })
        rows.append({"image_id": image_id, "predictions": preds})
        if (i + 1) % 20 == 0 or i == len(images) - 1:
            print("  %d/%d  last %s boxes=%d" %
                  (i + 1, len(images), image_id, len(preds)))

    if fail_read:
        raise RuntimeError(
            "image read failed, fix and rerun (must not fake empty): %s" %
            fail_read[:10])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("written %d lines -> %s" % (len(rows), output_path))
    return rows


def validate(output_path, image_dir=None):
    """对生成的 jsonl 做格式校验，返回问题列表（空列表表示通过）。"""
    problems = []
    ids = []
    with open(output_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except Exception as e:
                problems.append("line %d: invalid JSON %s" % (ln, e))
                continue
            if set(row.keys()) != {"image_id", "predictions"}:
                problems.append("line %d: top keys must be image_id,predictions"
                                % ln)
            iid = row.get("image_id")
            if not isinstance(iid, str) or not iid:
                problems.append("line %d: bad image_id" % ln)
            ids.append(iid)
            preds = row.get("predictions")
            if not isinstance(preds, list):
                problems.append("line %d: predictions not list" % ln)
                continue
            for k, p in enumerate(preds):
                if set(p.keys()) != {"points", "transcription",
                                     "confidence"}:
                    problems.append("line %d pred %d: bad keys" % (ln, k))
                pts = p.get("points")
                if not (isinstance(pts, list) and len(pts) == 8 and
                        all(is_finite_num(v) for v in pts)):
                    problems.append("line %d pred %d: bad points" % (ln, k))
                t = p.get("transcription")
                if not isinstance(t, str):
                    problems.append("line %d pred %d: transcription not str"
                                    % (ln, k))
                c = p.get("confidence")
                if not (is_finite_num(c) and 0.0 <= float(c) <= 1.0):
                    problems.append("line %d pred %d: confidence not in [0,1]"
                                    % (ln, k))
    # 重复 / 与图片目录一一对应
    dup = {x for x in ids if ids.count(x) > 1}
    if dup:
        problems.append("duplicate image_id: %s" % list(dup)[:5])
    if image_dir:
        disk = {os.path.basename(p) for p in list_images(image_dir)}
        only_json = set(ids) - disk
        only_disk = disk - set(ids)
        if only_json:
            problems.append("in json but not on disk: %s" % list(only_json)[:5])
        if only_disk:
            problems.append("on disk but not in json: %s" % list(only_disk)[:5])
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output", default=os.path.join(PROJ, "predictions.jsonl"))
    ap.add_argument("--det-weights", default=None)
    ap.add_argument("--rec-weights", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--limit-side-len", type=int, default=960)
    args = ap.parse_args()

    from ocr_engine import DET_WEIGHTS, REC_WEIGHTS
    detw = args.det_weights or DET_WEIGHTS
    recw = args.rec_weights or REC_WEIGHTS

    generate(args.image_dir, args.output, detw, recw,
             limit=args.limit, limit_side_len=args.limit_side_len)
    # limit 仅处理子集，不做与整个目录的一一对应校验，只校验已写行的格式；
    # 正式全量生成（无 limit）才校验图片与行一一对应。
    correspondence_dir = None if args.limit else args.image_dir
    problems = validate(args.output, correspondence_dir)
    if problems:
        print("VALIDATION FAILED:")
        for p in problems[:30]:
            print("  -", p)
        sys.exit(1)
    print("VALIDATION PASSED: format OK, image correspondence OK.")


if __name__ == "__main__":
    main()
