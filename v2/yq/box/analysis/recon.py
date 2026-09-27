"""3D model: refined hull mesh -> smoothing -> decimation -> vertex colours -> metric GLB + previews.

GLB convention (glTF 2.0): metres, Y up. Object frame (mm, Z up) -> glTF: (x, z, -y) / 1000.
The origin stays at the platter centre, so the model stands on y=0 at its real size.
Vertex colours come from the photos that see each vertex best (visibility by z-buffer),
weighted by how frontal the view is. SfM/MVS with pycolmap was evaluated and NOT used: its dense
MVS needs CUDA (absent on the Mac and not worth it on the Jetson) and, with calibrated poses, the
silhouette hull + photo-consistency carving already gives the geometry (docs/box_analysis.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .geometry import unit
from .render import preview, surface_samples, visibility, zbuffer

TARGET_FACES = 150_000


def clean_mesh(verts: np.ndarray, faces: np.ndarray, target_faces: int = TARGET_FACES, smooth_iters: int = 8):
    """Taubin smoothing (volume preserving) + quadric decimation. Returns a trimesh.Trimesh."""
    import trimesh
    from trimesh import smoothing
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    m.remove_unreferenced_vertices()
    if smooth_iters > 0:
        smoothing.filter_taubin(m, lamb=0.5, nu=0.53, iterations=smooth_iters)
    if len(m.faces) > target_faces:
        try:
            m = m.simplify_quadric_decimation(face_count=int(target_faces))
        except Exception:
            pass   # fast-simplification missing: keep the dense mesh
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: len(p.faces))
    m.fix_normals()
    return m


def vertex_colors(verts: np.ndarray, normals: np.ndarray, faces: np.ndarray, photos: list, top_k: int = 3) -> np.ndarray:
    """photos: [(Camera, RGB uint8 image)]. Returns (N,4) uint8 RGBA."""
    import cv2
    n = len(verts)
    acc = np.zeros((n, 3), np.float64)
    wsum = np.zeros(n, np.float64)
    best = np.zeros((n, top_k), np.float64)          # keep only the top_k weights per vertex (sharper texture)
    pts = surface_samples(verts, faces)
    W = []
    for cam, _img in photos:
        zb = zbuffer(cam, pts, scale=0.5)
        vis = visibility(cam, verts, normals, zb, 0.5, tol=2.0, min_cos=0.2)
        c = np.einsum("ij,ij->i", normals, unit(cam.center[None] - verts))
        W.append(np.where(vis, np.clip(c, 0, 1) ** 4, 0.0))
    W = np.array(W)                                  # (V, N)
    if len(W):
        kth = np.sort(W, axis=0)[-min(top_k, len(W))]
        W = np.where(W >= kth[None] - 1e-12, W, 0.0)
    for (cam, img), w in zip(photos, W):
        sel = np.nonzero(w > 0)[0]
        if not len(sel):
            continue
        uv = cam.project(verts[sel]).astype(np.float32)
        col = cv2.remap(img, uv[:, 0].reshape(1, -1), uv[:, 1].reshape(1, -1), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).reshape(-1, 3)
        acc[sel] += w[sel, None] * col
        wsum[sel] += w[sel]
    rgb = np.where(wsum[:, None] > 0, acc / np.maximum(wsum[:, None], 1e-9), 110.0)
    return np.concatenate([np.clip(rgb, 0, 255), np.full((n, 1), 255.0)], axis=1).astype(np.uint8)


def to_gltf_frame(verts_mm: np.ndarray) -> np.ndarray:
    return np.stack([verts_mm[:, 0], verts_mm[:, 2], -verts_mm[:, 1]], axis=1) / 1000.0


def export_glb(mesh, colors: np.ndarray, path: Path) -> Path:
    import trimesh
    g = trimesh.Trimesh(vertices=to_gltf_frame(np.asarray(mesh.vertices)), faces=np.asarray(mesh.faces),
                        vertex_colors=colors, process=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = g.export(file_type="glb")
    path.write_bytes(data)
    return path


def reconstruct(verts: np.ndarray, faces: np.ndarray, photos: list, out_dir: Path, previews: int = 4,
                target_faces: int = TARGET_FACES) -> dict:
    """Build model.glb + preview PNGs in out_dir. photos: [(Camera, RGB image)] (any resolution the cameras match)."""
    import cv2
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = clean_mesh(verts, faces, target_faces)
    V = np.asarray(m.vertices)
    F = np.asarray(m.faces)
    N = unit(np.asarray(m.vertex_normals))
    cols = vertex_colors(V, N, F, photos)
    glb = export_glb(m, cols, out_dir / "model.glb")
    arts = {"model_glb": glb.name}
    for k in range(previews):
        az = -90.0 + 360.0 * k / previews
        img = preview(V, F, cols[:, :3], N, az, 25.0)
        name = "model_view%d.png" % k
        cv2.imwrite(str(out_dir / name), img[:, :, ::-1])
        arts["model_view%d" % k] = name
    info = {"faces": int(len(F)), "vertices": int(len(V)), "watertight": bool(m.is_watertight),
            "volume_cm3": float(abs(m.volume) / 1000.0) if m.is_watertight else None, "glb_bytes": glb.stat().st_size}
    return {"artifacts": arts, "info": info, "mesh": m, "colors": cols}
