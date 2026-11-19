# -*- coding: utf-8 -*-
"""
OCR 模型评估（在 valid 集上），对应任务书第 4.4 节三项指标：
- e2e : 端到端 F1（主指标）。检测框 IoU>=0.5 且文本 NFC 规范化后完全一致才匹配，
         图内一对一最大匹配，全局累计 TP/FP/FN。
- det : 检测 precision / recall / F1（只看四边形 IoU>=0.5）。
- rec : 识别 line_acc（整句准确率）与 CER（在 GT 裁剪文本行上，排除检测误差）。

用法：
  .venv\\Scripts\\python.exe deploy\\evaluate.py --mode all
  .venv\\Scripts\\python.exe deploy\\evaluate.py --mode e2e --det-weights models/det_output/latest --rec-weights models/rec_output/latest
"""
import argparse
import json
import os
import sys
import time
import unicodedata

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_engine import (  # noqa: E402
    PROJ, OCREngine, EagerTextDetector, EagerTextRecognizer, imread_unicode)
from shapely.geometry import Polygon  # noqa: E402

IMG_DIR = os.path.join(PROJ, "data", "train", "img")
DET_VALID = os.path.join(PROJ, "data", "ppocr", "det", "det_gt_valid.txt")
REC_VALID = os.path.join(PROJ, "data", "ppocr", "rec", "gt", "rec_gt_valid.txt")
REC_GT_DIR = os.path.join(PROJ, "data", "ppocr", "rec", "gt")


def norm_text(s):
    return unicodedata.normalize("NFC", s).strip()


def _poly(p):
    return Polygon(np.asarray(p, dtype=np.float64)).buffer(0)


def poly_iou(p1, p2):
    try:
        g1, g2 = _poly(p1), _poly(p2)
        if g1.is_empty or g2.is_empty or not g1.is_valid or not g2.is_valid:
            return 0.0
        inter = g1.intersection(g2).area
        union = g1.area + g2.area - inter
        return float(inter) / float(union) if union > 0 else 0.0
    except Exception:
        return 0.0


def overlap_ratio_on_pred(pred, gt):
    try:
        gp, gg = _poly(pred), _poly(gt)
        if gp.is_empty or gp.area <= 0:
            return 0.0
        return float(gp.intersection(gg).area) / float(gp.area)
    except Exception:
        return 0.0


def max_match(condition, iou):
    """二分图一对一最大匹配。condition/iou 形状 [n_gt, n_pred]。"""
    n_gt, n_pred = condition.shape
    if n_gt == 0 or n_pred == 0:
        return []
    cost = np.where(condition, 1.0 - iou, 1e6)
    ri, ci = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(ri, ci) if condition[r, c]]


def read_label_lines(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


def parse_det_line(line):
    name, js = line.split("\t")[:2]
    gts = json.loads(js)
    valid, ignore = [], []
    for g in gts:
        pts = np.asarray(g["points"], dtype=np.float64)
        if g.get("ignore") in (1, True, "1") or \
                not norm_text(g.get("transcription", "")):
            ignore.append(pts)
        else:
            valid.append((pts, norm_text(g.get("transcription", ""))))
    return name, valid, ignore


def prf1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def match_counts(gt_valid, gt_ignore, pred_items, require_text):
    """
    pred_items: list of (pts, text)
    require_text=True  -> e2e（IoU>=0.5 且文本一致）
    require_text=False -> det（只看 IoU）
    返回 (TP, FP, FN, ignored_fp)。
    """
    n_gt = len(gt_valid)
    n_pr = len(pred_items)
    iou = np.zeros((n_gt, n_pr), dtype=np.float64)
    cond = np.zeros((n_gt, n_pr), dtype=bool)
    for gi, (gpts, gtext) in enumerate(gt_valid):
        for pi, (ppts, ptext) in enumerate(pred_items):
            v = poly_iou(gpts, ppts)
            iou[gi, pi] = v
            if v >= 0.5 and (not require_text or norm_text(ptext) == gtext):
                cond[gi, pi] = True
    pairs = max_match(cond, iou)
    mg = {r for r, _ in pairs}
    mp = {c for _, c in pairs}
    tp = len(pairs)
    fp = 0
    ignored = 0
    for pi, (ppts, _) in enumerate(pred_items):
        if pi in mp:
            continue
        if any(overlap_ratio_on_pred(ppts, ig) > 0.5 for ig in gt_ignore):
            ignored += 1
        else:
            fp += 1
    fn = n_gt - len(mg)
    return tp, fp, fn, ignored


def load_valid_images(lines, max_images):
    out = []
    for line in lines:
        name, valid, ignore = parse_det_line(line)
        path = name if os.path.isabs(name) else os.path.join(IMG_DIR, name)
        img = imread_unicode(path)
        if img is None:
            continue
        out.append((name, img, valid, ignore))
        if max_images and len(out) >= max_images:
            break
    return out


def eval_det(det_weights, max_images, limit_side_len):
    print("=" * 60)
    print("[DET] loading:", det_weights)
    detector = EagerTextDetector(det_weights=det_weights,
                                 limit_side_len=limit_side_len)
    samples = load_valid_images(read_label_lines(DET_VALID), max_images)
    TP = FP = FN = 0
    t0 = time.time()
    for i, (name, img, valid, ignore) in enumerate(samples):
        boxes = detector(img)
        items = [(b, "") for b in boxes]
        tp, fp, fn, _ = match_counts(valid, ignore, items, require_text=False)
        TP += tp
        FP += fp
        FN += fn
        if (i + 1) % 25 == 0:
            print("  %d/%d TP=%d FP=%d FN=%d (%.1fs)" %
                  (i + 1, len(samples), TP, FP, FN, time.time() - t0))
    p, r, f = prf1(TP, FP, FN)
    print("-" * 60)
    print("[DET] images=%d TP=%d FP=%d FN=%d" % (len(samples), TP, FP, FN))
    print("[DET] precision=%.4f recall=%.4f F1=%.4f time=%.1fs" %
          (p, r, f, time.time() - t0))
    return {"precision": p, "recall": r, "f1": f,
            "TP": TP, "FP": FP, "FN": FN}


def eval_e2e(det_weights, rec_weights, max_images, limit_side_len):
    print("=" * 60)
    print("[E2E] loading det+rec")
    engine = OCREngine(det_weights=det_weights, rec_weights=rec_weights,
                       limit_side_len=limit_side_len)
    samples = load_valid_images(read_label_lines(DET_VALID), max_images)
    TP = FP = FN = 0
    ignored = 0
    t0 = time.time()
    for i, (name, img, valid, ignore) in enumerate(samples):
        boxes, texts, scores = engine(img, keep_all=True)
        items = list(zip(boxes, texts))
        tp, fp, fn, ig = match_counts(valid, ignore, items, require_text=True)
        TP += tp
        FP += fp
        FN += fn
        ignored += ig
        if (i + 1) % 25 == 0:
            print("  %d/%d TP=%d FP=%d FN=%d (%.1fs)" %
                  (i + 1, len(samples), TP, FP, FN, time.time() - t0))
    p, r, f = prf1(TP, FP, FN)
    print("-" * 60)
    print("[E2E] images=%d TP=%d FP=%d FN=%d (ignored-FP=%d)" %
          (len(samples), TP, FP, FN, ignored))
    print("[E2E] precision=%.4f recall=%.4f F1=%.4f time=%.1fs" %
          (p, r, f, time.time() - t0))
    return {"precision": p, "recall": r, "f1": f,
            "TP": TP, "FP": FP, "FN": FN, "ignored_fp": ignored}


def edit_distance(a, b):
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def eval_rec(rec_weights, max_samples, batch_num):
    print("=" * 60)
    print("[REC] loading:", rec_weights)
    recognizer = EagerTextRecognizer(rec_weights=rec_weights,
                                     rec_batch_num=batch_num)
    lines = read_label_lines(REC_VALID)
    if max_samples and max_samples < len(lines):
        lines = lines[:max_samples]
    exact = 0
    total_edit = 0
    total_chars = 0
    t0 = time.time()
    BATCH = 32
    done = 0
    for start in range(0, len(lines), BATCH):
        crops, gts = [], []
        for line in lines[start:start + BATCH]:
            rel, gt = line.split("\t")[:2]
            cp = rel if os.path.isabs(rel) else \
                os.path.normpath(os.path.join(REC_GT_DIR, rel))
            img = imread_unicode(cp)
            if img is None:
                continue
            crops.append(img)
            gts.append(norm_text(gt))
        if not crops:
            continue
        res = recognizer(crops)
        for (pred, _s), gt in zip(res, gts):
            pred = norm_text(pred)
            if pred == gt:
                exact += 1
            total_edit += edit_distance(pred, gt)
            total_chars += len(gt)
        done += len(crops)
        if (start // BATCH + 1) % 5 == 0:
            print("  %d/%d acc=%.4f (%.1fs)" %
                  (done, len(lines), exact / max(done, 1), time.time() - t0))
    acc = exact / len(lines) if lines else 0.0
    cer = total_edit / total_chars if total_chars > 0 else 0.0
    print("-" * 60)
    print("[REC] samples=%d exact=%d line_acc=%.4f CER=%.4f time=%.1fs" %
          (len(lines), exact, acc, cer, time.time() - t0))
    return {"line_acc": acc, "cer": cer, "exact": exact, "n": len(lines)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["det", "rec", "e2e", "all"],
                    default="all")
    ap.add_argument("--det-weights", default=None)
    ap.add_argument("--rec-weights", default=None)
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--batch-num", type=int, default=6)
    ap.add_argument("--limit-side-len", type=int, default=960)
    args = ap.parse_args()

    from ocr_engine import DET_WEIGHTS, REC_WEIGHTS
    detw = args.det_weights or DET_WEIGHTS
    recw = args.rec_weights or REC_WEIGHTS

    result = {}
    if args.mode in ("det", "all"):
        result["det"] = eval_det(detw, args.max_images, args.limit_side_len)
    if args.mode in ("rec", "all"):
        result["rec"] = eval_rec(recw, args.max_samples, args.batch_num)
    if args.mode in ("e2e", "all"):
        result["e2e"] = eval_e2e(detw, recw, args.max_images,
                                 args.limit_side_len)
    print("=" * 60)
    print("SUMMARY:", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
