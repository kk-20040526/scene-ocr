# -*- coding: utf-8 -*-
"""
OCR 本地推理后端（FastAPI）

启动：
  set OMP_NUM_THREADS=8
  .venv\\Scripts\\python.exe -m uvicorn deploy.server:app --host 0.0.0.0 --port 8000
  （在项目根目录下执行）

接口：
  GET  /health  健康检查
  POST /ocr      multipart 上传一张图片，返回：
                 - annotated_image: 叠加绿色文字框的 JPEG（base64）
                 - results: 每个文字行 {points(8), text, score}
"""
import base64
import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_engine import OCREngine  # noqa: E402

app = FastAPI(title="OCR 光学字符识别系统")
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = OCREngine()
    return _engine


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "empty file"}, status_code=400)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "cannot decode image"}, status_code=400)

    engine = get_engine()
    boxes, texts, scores = engine(img, keep_all=True)

    vis = img.copy()
    results = []
    for box, text, score in zip(boxes, texts, scores):
        box = np.asarray(box, dtype=np.float64)
        pts = np.round(box).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        results.append({
            "points": [int(round(v)) for v in box.reshape(-1).tolist()],
            "text": text if isinstance(text, str) else "",
            "score": round(float(score), 4),
        })

    ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    return {
        "image_id": file.filename,
        "count": len(results),
        "results": results,
        "annotated_image": annotated,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
