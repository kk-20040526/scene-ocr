# -*- coding: utf-8 -*-
"""在 valid 图上用引擎推理，输出纯绿框（不写中文标签）样例图与文本清单，
供报告第4章挑选 正确/检测错误/识别错误 样例。训练后运行：
  python deploy/generate_sample_figures.py \
    --det-weights models/det_output/latest --rec-weights models/rec_output/latest --n 16
"""
import os
import sys
import json
import argparse
import time
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from ocr_engine import OCREngine, imread_unicode, imwrite_unicode  # noqa: E402


def load_valid_names():
    sp = os.path.join(PROJ, 'data', 'ppocr', 'split.json')
    img_dir = os.path.join(PROJ, 'data', 'train', 'img')
    names = []
    if os.path.exists(sp):
        with open(sp, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if 'valid' in data and isinstance(data['valid'], list):
                names = list(data['valid'])
            else:
                names = [k for k, v in data.items() if str(v).lower() == 'valid']
        elif isinstance(data, list):
            names = [x.get('image') if isinstance(x, dict) else x for x in data
                     if str(x.get('split', '') if isinstance(x, dict) else '').lower() == 'valid']
    paths = []
    for n in names:
        name = os.path.basename(n)
        if not os.path.splitext(name)[1]:
            name = name + '.jpg'
        paths.append(os.path.join(img_dir, name))
    paths = [p for p in paths if os.path.exists(p)]
    return paths


def draw_green_boxes(img, boxes):
    vis = img.copy()
    for box in boxes:
        pts = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 200, 0), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det-weights', default=os.path.join(PROJ, 'models', 'ch_PP-OCRv4_det_train', 'best_accuracy'))
    ap.add_argument('--rec-weights', default=os.path.join(PROJ, 'models', 'ch_PP-OCRv3_rec_train', 'student'))
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--out', default=os.path.join(PROJ, 'deploy', 'sample_figs'))
    ap.add_argument('--start', type=int, default=0, help='从 valid 列表第几张开始（便于挑不同样例）')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    eng = OCREngine(det_weights=args.det_weights, rec_weights=args.rec_weights)

    valid = load_valid_names()
    print('valid images:', len(valid))
    sel = valid[args.start:args.start + args.n]
    manifest = []
    for i, p in enumerate(sel):
        img = imread_unicode(p)
        if img is None:
            continue
        t = time.time()
        boxes, texts, scores = eng(img)
        dt = time.time() - t
        vis = draw_green_boxes(img, boxes)
        out_name = 'sample_%02d_%s' % (i, os.path.basename(p))
        imwrite_unicode(os.path.join(args.out, out_name), vis)
        item = {'file': out_name, 'src': os.path.basename(p), 'n': len(boxes),
                'time_s': round(dt, 2),
                'lines': [{'text': t, 'score': round(float(s), 3)} for t, s in zip(texts, scores)]}
        manifest.append(item)
        print('%s boxes=%d time=%.2fs' % (out_name, len(boxes), dt))
        for ln in item['lines']:
            print('    [%.2f] %s' % (ln['score'], ln['text']))
    with open(os.path.join(args.out, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print('sample figs ->', args.out)


if __name__ == '__main__':
    main()
