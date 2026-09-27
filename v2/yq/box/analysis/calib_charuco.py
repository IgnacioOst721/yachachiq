"""ChArUco camera intrinsics and turntable extrinsics (OpenCV >= 4.7 objdetect aruco API).

Verified against the installed OpenCV (5.0): cv2.aruco.CharucoBoard((cols, rows), square, marker,
dictionary, ids), CharucoDetector(board).detectBoard(img) -> (charucoCorners, charucoIds,
markerCorners, markerIds), board.matchImagePoints(corners, ids) -> (objPoints, imgPoints).
Board frame: origin at the top-left corner of the printed board, x right, y down (as printed),
z = x cross y (into the table when the board lies face up), millimetres.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import Camera, rodrigues, rotvec, rotz, unit


@dataclass
class BoardSpec:
    cols: int
    rows: int
    square_mm: float
    marker_mm: float
    dictionary: str = "DICT_5X5_100"
    first_id: int = 0
    name: str = "board"

    def board(self):
        import cv2
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        n = (self.cols * self.rows) // 2
        return cv2.aruco.CharucoBoard((self.cols, self.rows), float(self.square_mm), float(self.marker_mm), d,
                                      np.arange(self.first_id, self.first_id + n, dtype=np.int32))

    @property
    def size_mm(self) -> tuple:
        return self.cols * self.square_mm, self.rows * self.square_mm

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# Handheld board for intrinsics (fits inside the 300 mm box) and the flat board for the platter
# (108 mm square: diagonal 153 mm < platter 180 mm). Different marker ids so they never mix up.
INTRINSICS_BOARD = BoardSpec(8, 6, 18.0, 13.5, "DICT_5X5_100", 0, "intrinsics")
TURNTABLE_BOARD = BoardSpec(6, 6, 18.0, 13.5, "DICT_5X5_100", 50, "turntable")


def detect(image, spec: BoardSpec, min_corners: int = 6) -> Optional[tuple]:
    """-> (obj_pts (N,3) mm, img_pts (N,2) px, ids (N,)) or None."""
    import cv2
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    board = spec.board()
    det = cv2.aruco.CharucoDetector(board)
    cc, ci, _mc, _mi = det.detectBoard(gray)
    if cc is None or ci is None or len(ci) < min_corners:
        return None
    obj, img = board.matchImagePoints(cc, ci)
    if obj is None or len(obj) < min_corners:
        return None
    return obj.reshape(-1, 3).astype(np.float64), img.reshape(-1, 2).astype(np.float64), ci.ravel().astype(int)


def calibrate_intrinsics(images: list, spec: BoardSpec = INTRINSICS_BOARD, fix_k3: bool = True) -> dict:
    """Camera matrix + distortion from several photos of the handheld board at varied tilts."""
    import cv2
    objs, imgs, used = [], [], 0
    size = None
    for im in images:
        size = (im.shape[1], im.shape[0])
        r = detect(im, spec)
        if r is None:
            continue
        objs.append(r[0].astype(np.float32))
        imgs.append(r[1].astype(np.float32))
        used += 1
    if used < 4:
        raise ValueError("solo %d fotos con el tablero detectado; se necesitan al menos 4 (ideal 15-25)" % used)
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, size, None, None, flags=flags)
    per = []
    for o, i, rv, tv in zip(objs, imgs, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, rv, tv, K, dist)
        per.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i) ** 2, axis=1)))))
    return {"K": K.tolist(), "dist": dist.ravel()[:5].tolist(), "width": size[0], "height": size[1], "rms_px": float(rms),
            "n_images": used, "per_image_rms_px": per, "source": "charuco", "board": spec.to_dict()}


def _pnp(obj, img, cam: Camera) -> tuple:
    import cv2
    ok, rv, tv = cv2.solvePnP(obj, img, cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rv, tv = cv2.solvePnP(obj, img, cam.K, cam.dist)
    return rodrigues(rv.ravel()), tv.ravel()


def _project(cam_K, cam_dist, R, t, X):
    c = Camera("x", cam_K, R, t, 1, 1, cam_dist)
    return c.project(X)


R_FLIP = np.diag([1.0, -1.0, -1.0])   # board lying face up: board x = world x, board y = -world y, board z = -world z


def _init_world(obs: list, direction_hint: Optional[int], thickness: float) -> tuple:
    """Initial world frame in the first camera's coordinates from its PnP poses.
    obs: [(deg, R_cb, t_cb)] of ONE camera. Returns (R_cw, t_cw, direction, (bx, by))."""
    obs = sorted(obs, key=lambda o: abs(((o[0] + 180) % 360) - 180))
    d0, R0, t0 = obs[0]
    acc = np.zeros(3)
    for d, R, _t in obs[1:]:
        dd = ((d - d0 + 180) % 360) - 180
        if abs(dd) < 10:
            continue
        acc += math.copysign(1.0, dd) * rotvec(R @ R0.T)
    if np.linalg.norm(acc) < 1e-9:
        raise ValueError("las vistas del tablero no cubren suficientes ángulos del plato")
    a_signed = unit(acc)
    up = -unit(R0[:, 2])                                  # board z points down into the platter
    direction = int(np.sign(a_signed @ up)) if direction_hint is None else int(direction_hint)
    z = up
    P = np.array([t for _d, _R, t in obs])
    e1 = unit(np.cross(z, [1.0, 0, 0]) if abs(z[0]) < 0.9 else np.cross(z, [0, 1.0, 0]))
    e2 = np.cross(z, e1)
    uv = np.stack([P @ e1, P @ e2], axis=1)
    A = np.column_stack([2 * uv, np.ones(len(uv))])      # Kasa circle fit
    sol, *_ = np.linalg.lstsq(A, np.sum(uv ** 2, axis=1), rcond=None)
    c = sol[0] * e1 + sol[1] * e2 + (P @ z).mean() * z
    x0 = rodrigues(-direction * math.radians(d0) * z) @ R0[:, 0]
    x = unit(x0 - (x0 @ z) * z)
    y = np.cross(z, x)
    R_cw = np.column_stack([x, y, z])                    # world axes in camera coords
    t_cw = c - thickness * z
    b = rotz(-direction * d0) @ (R_cw.T @ (t0 - t_cw))
    return R_cw, t_cw, direction, (float(b[0]), float(b[1]))


def calibrate_turntable(views: list, intrinsics: dict, spec: BoardSpec = TURNTABLE_BOARD, thickness_mm: float = 0.3,
                        direction: Optional[int] = None, refine_intrinsics: bool = False) -> dict:
    """Pose of every camera relative to the platter axis from photos of the flat board at known angles.

    views: [(camera_name, platter_deg, image)], at least ~8 angles over 360 deg per camera.
    intrinsics: name -> Camera (pose ignored) or {K, dist, width, height}.
    Returns {"cameras": {name: {R, t, rms_px, n}}, "direction", "rms_px", "axis_tilt_deg",
             "angle_scale", "board_xy_mm", "n_views"}.
    """
    from scipy.optimize import least_squares
    cams = {}
    for n, v in intrinsics.items():
        cams[n] = v if isinstance(v, Camera) else Camera(n, v["K"], np.eye(3), np.zeros(3), v["width"], v["height"], v.get("dist"))
    obs = {}
    for name, deg, img in views:
        r = detect(img, spec)
        if r is None:
            continue
        cam = cams[name].scaled(img.shape[1], img.shape[0]) if (img.shape[1], img.shape[0]) != (cams[name].width, cams[name].height) else cams[name]
        cams[name] = cam
        R, t = _pnp(r[0], r[1], cam)
        obs.setdefault(name, []).append((float(deg), r[0], r[1], R, t))
    if not obs:
        raise ValueError("no se detectó el tablero en ninguna foto")
    names = sorted(obs, key=lambda n: -len(obs[n]))
    ref = names[0]
    R_cw, t_cw, direction, (bx, by) = _init_world([(d, R, t) for d, _o, _i, R, t in obs[ref]], direction, thickness_mm)
    R_b0 = R_FLIP
    init = {}
    for n in names:
        if n == ref:
            init[n] = (R_cw.T, -R_cw.T @ t_cw)
            continue
        Ms, ts = [], []
        for d, _o, _i, R, t in obs[n]:
            Rw = rotz(direction * d) @ R_b0
            Rc = R @ Rw.T
            Ms.append(Rc)
            ts.append(t - Rc @ (rotz(direction * d) @ np.array([bx, by, thickness_mm])))
        U, _s, Vt = np.linalg.svd(np.sum(Ms, axis=0))
        Rc = U @ Vt
        init[n] = (Rc, np.mean(ts, axis=0))
    # world->camera for the reference camera: X_c = R_cw X_w + t_cw  (R_cw columns = world axes)
    init[ref] = (R_cw, t_cw)

    def unpack(p):
        out, k = {}, 0
        for n in names:
            out[n] = (rodrigues(p[k:k + 3]), p[k + 3:k + 6])
            k += 6
        wx, wy, bxx, byy, scale = p[k:k + 5]
        k += 5
        intr = {}
        if refine_intrinsics:
            for n in names:
                intr[n] = p[k:k + 6]
                k += 6
        return out, rodrigues([wx, wy, 0.0]) @ R_b0, np.array([bxx, byy, thickness_mm]), scale, intr

    p0 = []
    for n in names:
        p0 += list(rotvec(init[n][0])) + list(init[n][1])
    p0 += [0.0, 0.0, bx, by, 1.0]
    if refine_intrinsics:
        for n in names:
            K = cams[n].K
            p0 += [K[0, 0], K[1, 1], K[0, 2], K[1, 2], cams[n].dist[0], cams[n].dist[1]]

    def residuals(p):
        poses, Rb, b, scale, intr = unpack(p)
        res = []
        for n in names:
            K, dist = cams[n].K.copy(), cams[n].dist.copy()
            if refine_intrinsics:
                fx, fy, cx, cy, k1, k2 = intr[n]
                K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
                dist[0], dist[1] = k1, k2
            Rc, tc = poses[n]
            for d, o, i, _R, _t in obs[n]:
                Xw = (o @ Rb.T + b) @ rotz(direction * d * scale).T
                res.append((_project(K, dist, Rc, tc, Xw) - i).ravel())
        return np.concatenate(res)

    sol = least_squares(residuals, np.array(p0), loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=200)
    poses, Rb, b, scale, intr = unpack(sol.x)
    r = residuals(sol.x).reshape(-1, 2)
    out, k = {}, 0
    for n in names:
        m = sum(len(o[1]) for o in obs[n])
        e = np.sqrt(np.mean(np.sum(r[k:k + m] ** 2, axis=1)))
        k += m
        cam = {"R": poses[n][0].tolist(), "t": poses[n][1].tolist(), "rms_px": float(e), "n_views": len(obs[n])}
        if refine_intrinsics:
            fx, fy, cx, cy, k1, k2 = intr[n]
            cam["K"] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            cam["dist"] = [k1, k2] + list(cams[n].dist[2:])
        out[n] = cam
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, abs((Rb @ [0, 0, 1.0])[2])))))
    return {"cameras": out, "direction": direction, "rms_px": float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))),
            "axis_tilt_deg": tilt, "angle_scale": float(scale), "board_xy_mm": [float(b[0]), float(b[1])],
            "n_views": int(sum(len(v) for v in obs.values())), "source": "charuco-turntable", "board": spec.to_dict(),
            "thickness_mm": thickness_mm, "_image_sizes": {n: [cams[n].width, cams[n].height] for n in names}}
