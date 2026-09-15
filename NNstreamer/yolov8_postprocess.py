#!/usr/bin/env python
"""
YOLOv8 raw-head post-processing (pure numpy, no nnstreamer dependency).

Decodes the THREE raw INT8 heads from a raw-head YOLOv8 mtk_converter model:

    int8 [1, 64+nc, 80, 80] @ stride  8
    int8 [1, 64+nc, 40, 40] @ stride 16
    int8 [1, 64+nc, 20, 20] @ stride 32

Channel layout per head (from concat(cv2, cv3)):
    channels   0 .. 63       : box DFL logits, laid out coord*reg_max + bin,
                               coord order = [left, top, right, bottom], reg_max=16
    channels  64 .. 64+nc-1  : class logits (NO objectness in YOLOv8)

Decode (reproduces ultralytics Detect):
    dist_c   = sum_b softmax(box_logits[c])[b] * b        # per coord, grid units
    anchor   = (gx + 0.5, gy + 0.5)                        # cell centre, grid units
    x1,y1    = (anchor - (left,top)) * stride              # pixels @ 640 scale
    x2,y2    = (anchor + (right,bottom)) * stride
    score    = sigmoid(class_logits)                       # per class, no objectness

detect() returns (x1, y1, x2, y2, class_id, score) in 640x640 model space,
i.e. the SAME tuple format as yolov5_postprocess.detect(), so the demo only
needs its import swapped (and a matching label file).
"""

import numpy as np

REG_MAX = 16
BOX_CH = 4 * REG_MAX  # 64

# Number of classes for THIS model. Your bottle model reported output (1,18,8400)
# => nc = 18 - 4 = 14. Set to match your model / label file.
NC = 80

STRIDE = {80: 8, 40: 16, 20: 32}

# Per-head output dequantization (scale, zero_point), keyed by grid size.
# This model is FLOAT32 (get_output_details quant = (0.0, 0) => not quantized),
# so dequant is the identity: real = (v - 0) * 1.0 = v.
# If you later export an INT8 model, replace these with the real
# get_output_details()[i]['quantization'] values (scale, zero_point).
DEQUANT = {
    80: (1.0, 0),
    40: (1.0, 0),
    20: (1.0, 0),
}

_BINS = np.arange(REG_MAX, dtype=np.float32)


def _sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def decode_heads(input_arrays, nc=NC):
    """List of raw int8 head arrays (1-D, NCHW memory) -> (N, 4+nc) float32,
    where columns are [x1, y1, x2, y2, cls0..cls_{nc-1}] in 640 pixel space."""
    ch = BOX_CH + nc
    results = []
    for arr in input_arrays:
        arr = np.asarray(arr)
        n = arr.size
        grid = int(round((n / ch) ** 0.5))
        if grid not in STRIDE or ch * grid * grid != n:
            raise ValueError("unexpected head size %d (ch=%d grid=%d)" % (n, ch, grid))

        v = arr.astype(np.float32).reshape(ch, grid, grid)
        scale, zp = DEQUANT[grid]
        v = (v - zp) * scale
        stride = STRIDE[grid]

        # ---- box: DFL over 16 bins per coord ----
        box = v[:BOX_CH].reshape(4, REG_MAX, grid, grid)          # (coord, bin, ny, nx)
        prob = _softmax(box, axis=1)                              # softmax over bins
        dist = np.tensordot(_BINS, prob, axes=(0, 1))            # (4, ny, nx) grid units
        left, top, right, bottom = dist[0], dist[1], dist[2], dist[3]

        gx = np.arange(grid, dtype=np.float32).reshape(1, grid)   # x index over columns
        gy = np.arange(grid, dtype=np.float32).reshape(grid, 1)   # y index over rows
        ax = gx + 0.5
        ay = gy + 0.5

        x1 = (ax - left) * stride
        y1 = (ay - top) * stride
        x2 = (ax + right) * stride
        y2 = (ay + bottom) * stride

        cls = _sigmoid(v[BOX_CH:BOX_CH + nc])                     # (nc, ny, nx)

        out = np.empty((grid * grid, 4 + nc), dtype=np.float32)
        out[:, 0] = x1.reshape(-1)
        out[:, 1] = y1.reshape(-1)
        out[:, 2] = x2.reshape(-1)
        out[:, 3] = y2.reshape(-1)
        out[:, 4:] = cls.reshape(nc, -1).T
        results.append(out)

    return np.concatenate(results, axis=0)


def _nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def detect(input_arrays, conf_thres=0.25, iou_thres=0.45, max_det=300, nc=NC):
    """Decode + threshold + per-class NMS.
    Returns list of (x1, y1, x2, y2, class_id, score) in 640x640 model space."""
    d = decode_heads(input_arrays, nc=nc)
    boxes = d[:, :4]
    cls_scores = d[:, 4:]
    cls_id = cls_scores.argmax(axis=1)
    score = cls_scores[np.arange(cls_scores.shape[0]), cls_id]

    m = score >= conf_thres
    if not np.any(m):
        return []
    boxes = boxes[m]
    scr = score[m]
    cid = cls_id[m]

    out = []
    for c in np.unique(cid):
        sel = np.where(cid == c)[0]
        keep = _nms(boxes[sel], scr[sel], iou_thres)
        for k in keep:
            j = sel[k]
            out.append((float(boxes[j, 0]), float(boxes[j, 1]),
                        float(boxes[j, 2]), float(boxes[j, 3]),
                        int(c), float(scr[j])))
    out.sort(key=lambda t: t[5], reverse=True)
    return out[:max_det]
