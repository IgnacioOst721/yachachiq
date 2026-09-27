"""Minimal RTMW / RTMPose / YOLOX inference on ONNX Runtime (no rtmlib needed at runtime).

The pre/post-processing is ported from rtmlib 0.0.16 (Apache-2.0, github.com/Tau-J/rtmlib:
tools/pose_estimation/pre_processings.py, post_processings.py, object_detection/yolox.py) so
the Jetson does not need rtmlib's pip dependencies (it pulls the CPU `onnxruntime` and pip
OpenCV, which would shadow onnxruntime-gpu). A test checks our output equals rtmlib's.

Execution providers, best first:
  Jetson: TensorRT (FP16, engine cache in MODELS_DIR/sign/trt_cache) -> CUDA -> CPU
  Mac:    CoreML -> CPU  (measure both: CoreML is not always faster for these models)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def providers_for(backend: str, cache_dir: Optional[Path] = None) -> list:
    """backend: 'auto' | 'tensorrt' | 'cuda' | 'coreml' | 'cpu'."""
    import onnxruntime as ort
    avail = ort.get_available_providers()
    trt_opts = {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                "trt_timing_cache_enable": True, "trt_max_workspace_size": 1 << 30}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        trt_opts["trt_engine_cache_path"] = str(cache_dir)
        trt_opts["trt_timing_cache_path"] = str(cache_dir)
    order = {
        "auto": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"],
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        "cuda": ["CUDAExecutionProvider"],
        "coreml": ["CoreMLExecutionProvider"],
        "cpu": [],
    }[backend]
    out = []
    for p in order:
        if p in avail:
            out.append((p, trt_opts) if p == "TensorrtExecutionProvider" else p)
    out.append("CPUExecutionProvider")
    return out


def make_session(onnx_path: Path, backend: str = "auto", cache_dir: Optional[Path] = None, threads: int = 0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    so.log_severity_level = 3
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers_for(backend, cache_dir))


# --------------------------------------------------------------------------------------------
# Top-down pose (RTMPose / RTMW, SimCC heads)
# --------------------------------------------------------------------------------------------
def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.25) -> Tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], np.float32)
    scale = np.array([(x2 - x1) * padding, (y2 - y1) * padding], np.float32)
    return center, scale


def _warp_matrix(center: np.ndarray, scale: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    import cv2
    src_dir = np.array([0.0, scale[0] * -0.5])
    dst_dir = np.array([0.0, out_w * -0.5])
    src = np.zeros((3, 2), np.float32)
    dst = np.zeros((3, 2), np.float32)
    src[0] = center
    src[1] = center + src_dir
    d = src[0] - src[1]
    src[2] = src[1] + np.array([-d[1], d[0]])
    dst[0] = [out_w * 0.5, out_h * 0.5]
    dst[1] = dst[0] + dst_dir
    d = dst[0] - dst[1]
    dst[2] = dst[1] + np.array([-d[1], d[0]])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def pose_preprocess(img: np.ndarray, bbox, input_wh: Tuple[int, int]):
    """-> (1,3,H,W) float32 blob, center, scale (aspect-corrected)."""
    import cv2
    w, h = input_wh
    center, scale = bbox_xyxy2cs(np.asarray(bbox, np.float32), padding=1.25)
    aspect = w / h
    if scale[0] > scale[1] * aspect:
        scale = np.array([scale[0], scale[0] / aspect], np.float32)
    else:
        scale = np.array([scale[1] * aspect, scale[1]], np.float32)
    M = _warp_matrix(center, scale, w, h)
    crop = cv2.warpAffine(img, M, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    blob = ((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(blob, np.float32), center, scale


def simcc_decode(simcc_x: np.ndarray, simcc_y: np.ndarray, center, scale, input_wh, split_ratio: float = 2.0):
    """SimCC argmax decode -> keypoints (K,2) in image pixels, scores (K,) (mean of x/y maxima)."""
    n, k, _ = simcc_x.shape
    sx = simcc_x.reshape(n * k, -1)
    sy = simcc_y.reshape(n * k, -1)
    locs = np.stack([np.argmax(sx, 1), np.argmax(sy, 1)], -1).astype(np.float32)
    vals = 0.5 * (np.amax(sx, 1) + np.amax(sy, 1))
    locs[vals <= 0.0] = -1
    kp = locs.reshape(n, k, 2)[0] / split_ratio
    kp = kp / np.asarray(input_wh, np.float32) * scale + center - scale / 2
    return kp.astype(np.float32), vals.reshape(n, k)[0].astype(np.float32)


class TopDownPose:
    def __init__(self, onnx_path: Path, input_wh: Tuple[int, int], backend: str = "auto",
                 cache_dir: Optional[Path] = None, threads: int = 0):
        self.sess = make_session(onnx_path, backend, cache_dir, threads)
        self.input_wh = tuple(input_wh)
        self.inp = self.sess.get_inputs()[0].name
        self.providers = self.sess.get_providers()

    def __call__(self, img_bgr: np.ndarray, bbox) -> Tuple[np.ndarray, np.ndarray]:
        blob, c, s = pose_preprocess(img_bgr, bbox, self.input_wh)
        sx, sy = self.sess.run(None, {self.inp: blob})[:2]
        return simcc_decode(sx, sy, c, s, self.input_wh)


# --------------------------------------------------------------------------------------------
# YOLOX person detector (OpenMMLab Human-Art export, NMS inside the graph)
# --------------------------------------------------------------------------------------------
class PersonDetector:
    def __init__(self, onnx_path: Path, input_hw: Tuple[int, int] = (416, 416), backend: str = "auto",
                 cache_dir: Optional[Path] = None, score_thr: float = 0.5):
        self.sess = make_session(onnx_path, backend, cache_dir)
        self.input_hw = tuple(input_hw)
        self.inp = self.sess.get_inputs()[0].name
        self.score_thr = score_thr

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        """-> (N, 5) boxes x1,y1,x2,y2,score in image pixels, best first."""
        import cv2
        ih, iw = self.input_hw
        ratio = min(ih / img_bgr.shape[0], iw / img_bgr.shape[1])
        rs = cv2.resize(img_bgr, (int(img_bgr.shape[1] * ratio), int(img_bgr.shape[0] * ratio)),
                        interpolation=cv2.INTER_LINEAR)
        pad = np.full((ih, iw, 3), 114, np.uint8)
        pad[:rs.shape[0], :rs.shape[1]] = rs
        blob = np.ascontiguousarray(pad.transpose(2, 0, 1)[None], np.float32)
        outs = self.sess.run(None, {self.inp: blob})
        det = outs[0]
        if det.ndim == 3:
            det = det[0]
        if det.shape[-1] == 5:                   # end2end export: dets (N,5) [+ labels (N,)]
            labels = outs[1][0] if len(outs) > 1 else np.zeros(len(det))
            keep = (det[:, 4] >= self.score_thr) & (labels == 0)
            boxes = det[keep].copy()
        else:
            raise RuntimeError("unexpected YOLOX output shape %s (need the end2end export with NMS)" % (det.shape,))
        boxes[:, :4] /= ratio
        return boxes[np.argsort(-boxes[:, 4])]
