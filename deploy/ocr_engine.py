# -*- coding: utf-8 -*-
"""
动态图 OCR 引擎（DBNet 检测 + CRNN/CTC(PP-OCRv3 双头) 识别）

说明：
- 本环境 paddle 2.6.2 CPU 的 paddle.jit.save 对 PaddleOCR 模型导出 inference model
  会缺失权重文件（resnet18 等标准模型可正常导出，仅 PaddleOCR 自定义 backbone/neck 结构异常）。
  因此推理直接用【动态图】加载训练产出的 .pdparams，与训练配置中的模型结构完全一致，
  预处理 / 后处理直接复用 PaddleOCR 源码中的算子与后处理，保证结果与官方一致。
- 检测：DBNet（PPLCNetV3 + RSEFPN + DBHead），输出概率图 -> DBPostProcess 四点框
- 识别：PP-OCRv3（MobileNetV1Enhance + SVTR + CTC，SAR 头仅训练时辅助，eval 时只走 CTC）
- 全流程：det -> 四点透视校正 crop -> rec -> 文本与置信度
"""
import os
import sys
import math
import copy
import cv2
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PADDLEOCR_DIR = os.path.join(PROJ, "PaddleOCR")
sys.path.insert(0, PADDLEOCR_DIR)

import paddle  # noqa: E402
from ppocr.modeling.architectures import build_model  # noqa: E402
from ppocr.postprocess import build_post_process  # noqa: E402
from ppocr.utils.save_load import load_model  # noqa: E402
from ppocr.data import create_operators, transform  # noqa: E402
from tools.program import load_config, merge_config, ArgsParser  # noqa: E402

DET_WEIGHTS = os.path.join(
    PROJ, "models", "ch_PP-OCRv4_det_train", "best_accuracy")
REC_WEIGHTS = os.path.join(
    PROJ, "models", "ch_PP-OCRv3_rec_train", "student")
DET_CFG = os.path.join(PROJ, "configs", "det_train.yml")
REC_CFG = os.path.join(PROJ, "configs", "rec_train.yml")


def imread_unicode(path):
    """cv2.imread 中文路径兼容。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """cv2.imwrite 中文路径兼容。"""
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def _build_model(config_path, weights, model_type):
    """依据训练配置构建动态图模型并加载权重，返回 (model, config)。"""
    argv = ["-c", config_path, "-o",
            "Global.pretrained_model=" + weights]
    FLAGS = ArgsParser().parse_args(argv)
    config = merge_config(load_config(FLAGS.config), FLAGS.opt)
    # 相对路径的字典文件转成基于 PaddleOCR 源码目录的绝对路径
    g = config.get("Global", {})
    cdp = g.get("character_dict_path")
    if cdp and not os.path.isabs(cdp):
        cand = os.path.join(PADDLEOCR_DIR, cdp.replace("/", os.sep))
        if os.path.exists(cand):
            g["character_dict_path"] = cand
    post_process = build_post_process(config["PostProcess"], config["Global"])
    if model_type == "rec" and hasattr(post_process, "character"):
        # PP-OCRv3 双头需要注入两个头的分类维度
        char_num = len(getattr(post_process, "character"))
        config["Architecture"]["Head"]["out_channels_list"] = {
            "CTCLabelDecode": char_num,
            "SARLabelDecode": char_num + 2,
            "NRTRLabelDecode": char_num + 3,
        }
    model = build_model(config["Architecture"])
    load_model(config, model, model_type=config["Architecture"]["model_type"])
    model.eval()
    return model, config


class EagerTextDetector:
    def __init__(self, det_cfg=DET_CFG, det_weights=DET_WEIGHTS,
                 limit_side_len=960, thresh=0.3, box_thresh=0.6,
                 unclip_ratio=1.5, use_dilation=False, box_type="quad"):
        self.box_type = box_type
        self.limit_side_len = limit_side_len
        pre_list = [
            {"DetResizeForTest": {"limit_side_len": limit_side_len,
                                  "limit_type": "max"}},
            {"NormalizeImage": {"std": [0.229, 0.224, 0.225],
                                "mean": [0.485, 0.456, 0.406],
                                "scale": "1./255.", "order": "hwc"}},
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        self.preprocess_op = create_operators(pre_list)
        self.postprocess_op = build_post_process({
            "name": "DBPostProcess",
            "thresh": thresh,
            "box_thresh": box_thresh,
            "max_candidates": 1000,
            "unclip_ratio": unclip_ratio,
            "use_dilation": use_dilation,
            "score_mode": "fast",
            "box_type": box_type,
        })
        self.model, _ = _build_model(det_cfg, det_weights, "det")

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_boxes(self, dt_boxes, image_shape):
        h, w = image_shape[0:2]
        out = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            if self.box_type == "poly":
                box = self.clip_det_res(box, h, w)
            else:
                box = self.order_points_clockwise(box)
                box = self.clip_det_res(box, h, w)
                rw = int(np.linalg.norm(box[0] - box[1]))
                rh = int(np.linalg.norm(box[0] - box[3]))
                if rw <= 3 or rh <= 3:
                    continue
            out.append(box)
        return np.array(out) if out else np.array([])

    def __call__(self, img):
        ori_shape = img.shape
        data = {"image": img}
        batch = transform(data, self.preprocess_op)
        im, shape_list = batch
        if im is None:
            return np.array([])
        im = np.expand_dims(im, axis=0).copy()
        shape_list = np.expand_dims(shape_list, axis=0)
        with paddle.no_grad():
            out = self.model(paddle.to_tensor(im))
        if isinstance(out, dict):
            out = out["maps"]
        preds = {"maps": out.numpy()}
        post = self.postprocess_op(preds, shape_list)
        boxes = post[0]["points"]
        return self.filter_boxes(boxes, ori_shape)


class EagerTextRecognizer:
    def __init__(self, rec_cfg=REC_CFG, rec_weights=REC_WEIGHTS,
                 rec_batch_num=6, drop_score=0.5):
        self.rec_image_shape = [3, 48, 320]
        self.rec_batch_num = rec_batch_num
        self.drop_score = drop_score
        self.model, config = _build_model(rec_cfg, rec_weights, "rec")
        # 复用与训练一致的 CTC 解码（字典、空格）
        self.postprocess_op = build_post_process(
            config["PostProcess"], config["Global"])

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        imgW = int(imgH * max_wh_ratio)
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = min(imgW, int(math.ceil(imgH * ratio)))
        resized = cv2.resize(img, (resized_w, imgH))
        resized = resized.astype("float32").transpose(2, 0, 1) / 255.0
        resized -= 0.5
        resized /= 0.5
        padding = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding[:, :, 0:resized_w] = resized
        return padding

    def __call__(self, img_list):
        """输入 crop(bgr) 列表，返回 [(text, score), ...]。"""
        img_num = len(img_list)
        if img_num == 0:
            return []
        width_list = [im.shape[1] / float(im.shape[0]) for im in img_list]
        indices = np.argsort(np.array(width_list))
        rec_res = [("", 0.0)] * img_num
        for beg in range(0, img_num, self.rec_batch_num):
            end = min(img_num, beg + self.rec_batch_num)
            norm_batch, wh_ratio_list = [], []
            max_wh_ratio = self.rec_image_shape[2] / self.rec_image_shape[1]
            for ino in range(beg, end):
                h, w = img_list[indices[ino]].shape[:2]
                wh = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh)
                wh_ratio_list.append(wh)
            for ino in range(beg, end):
                norm = self.resize_norm_img(
                    img_list[indices[ino]], max_wh_ratio)
                norm_batch.append(norm[np.newaxis, :])
            batch = np.concatenate(norm_batch).copy()
            with paddle.no_grad():
                out = self.model(paddle.to_tensor(batch))
            if isinstance(out, dict):
                out = out.get("ctc", out)
            preds = out.numpy() if hasattr(out, "numpy") else out
            rec = self.postprocess_op(
                preds, return_word_box=False,
                wh_ratio_list=wh_ratio_list, max_wh_ratio=max_wh_ratio)
            for rno in range(len(rec)):
                rec_res[indices[beg + rno]] = rec[rno]
        return rec_res


def get_rotate_crop_image(img, points):
    """四点透视校正裁剪。"""
    assert len(points) == 4, "shape of points must be 4*2"
    crop_w = int(max(np.linalg.norm(points[0] - points[1]),
                     np.linalg.norm(points[2] - points[3])))
    crop_h = int(max(np.linalg.norm(points[0] - points[3]),
                     np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
    M = cv2.getPerspectiveTransform(np.float32(points), pts_std)
    dst = cv2.warpPerspective(
        img, M, (crop_w, crop_h),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if dst.shape[0] * 1.0 / dst.shape[1] >= 1.5:
        dst = np.rot90(dst)
    return dst


def sorted_boxes(dt_boxes):
    """从上到下、从左到右排序。"""
    num = dt_boxes.shape[0]
    boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(boxes)
    for i in range(num - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
            else:
                break
    return _boxes


class OCREngine:
    def __init__(self, det_weights=DET_WEIGHTS, rec_weights=REC_WEIGHTS,
                 drop_score=0.5, limit_side_len=960, **kw):
        self.det = EagerTextDetector(
            det_weights=det_weights, limit_side_len=limit_side_len, **kw)
        self.rec = EagerTextRecognizer(
            rec_weights=rec_weights, drop_score=drop_score)
        self.drop_score = drop_score

    def __call__(self, img, keep_all=False):
        """返回 (boxes, texts, scores)，boxes 为 [N,4,2] 原图坐标。"""
        boxes = self.det(img)
        if boxes is None or len(boxes) == 0:
            return [], [], []
        boxes = sorted_boxes(boxes)
        crops = [get_rotate_crop_image(img, copy.deepcopy(b)) for b in boxes]
        rec_res = self.rec(crops)
        fb, ft, fs = [], [], []
        for box, rr in zip(boxes, rec_res):
            text, score = rr[0], float(rr[1])
            if not keep_all and score < self.drop_score:
                continue
            fb.append(box)
            ft.append(text)
            fs.append(score)
        return fb, ft, fs


# ------------------------- 可视化（中文） -------------------------
def draw_results(img, boxes, texts, scores, box_color=(0, 0, 255),
                 font_path=None):
    """在图上绘制四点框与识别文本。"""
    vis = img.copy()
    for box, text, score in zip(boxes, texts, scores):
        pts = np.array(box, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, box_color, 2)
        x, y = int(box[0][0]), int(box[0][1])
        label = "%s %.2f" % (text, score)
        # 尝试黑体中文，找不到则退化为默认字体
        font = None
        if font_path and os.path.exists(font_path):
            font = cv2.freetype.createFreeType2()
            font.loadFontData(fontName=font_path, id=0)
        if font is not None:
            font.putText(vis, label, (x, max(y - 6, 12)),
                         fontHeight=18, color=(255, 0, 0),
                         thickness=-1, line_type=cv2.LINE_AA,
                         bottomLeftOrigin=True)
        else:
            cv2.putText(vis, label, (x, max(y - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return vis


if __name__ == "__main__":
    import time
    import glob
    eng = OCREngine()
    imgs = sorted(glob.glob(os.path.join(PROJ, "data", "train", "img", "*.jpg")))
    out_dir = os.path.join(PROJ, "deploy", "demo_out")
    os.makedirs(out_dir, exist_ok=True)
    for p in imgs[:5]:
        img = imread_unicode(p)
        t = time.time()
        boxes, texts, scores = eng(img)
        dt = time.time() - t
        print(os.path.basename(p), "boxes=%d time=%.2fs" % (len(boxes), dt))
        for b, tx, sc in zip(boxes, texts, scores):
            print("   [%.2f] %s" % (sc, tx))
        vis = draw_results(img, boxes, texts, scores)
        imwrite_unicode(os.path.join(out_dir, os.path.basename(p)), vis)
    print("demo out ->", out_dir)
