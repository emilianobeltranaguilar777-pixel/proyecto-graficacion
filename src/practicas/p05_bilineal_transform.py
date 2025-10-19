#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p05 — Transformaciones bilineales (imagen/video)
Controles: [1][2][3] modo · [v] VIDEO/IMAGEN · [s] snapshot · [o] abrir imagen · [q]/ESC salir
"""
import os, math
import cv2 as cv
import numpy as np

# --- registro (2 args) ---
try:
    from src.core.registry import register
except Exception:
    try:
        from core.registry import register
    except Exception:
        def register(*_, **__):
            def deco(fn): return fn
            return deco

# --- CamKit + fallback ---
_USE_CAMKIT = False
try:
    from src.core.camkit import open_camera, frame_stream
    _USE_CAMKIT = True
except Exception:
    try:
        from core.camkit import open_camera, frame_stream
        _USE_CAMKIT = True
    except Exception:
        pass

def _fallback_open(device="/dev/video2", w=640, h=480, fps=30):
    cap = cv.VideoCapture(device, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv.CAP_PROP_FPS,          fps)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir {device}")
    def stream():
        while True:
            ok, f = cap.read()
            if not ok: break
            yield ok, f
    return cap, stream()

# --- util geom ---
def _corners(w, h):
    return np.array([[0,w,w,0],[0,0,h,h],[1,1,1,1]], np.float32)

def _to33(M23):
    M = np.eye(3, dtype=np.float32); M[:2] = M23; return M

def _to23(M33):
    return M33[:2]

def _bbox_for(M33, w, h):
    tc = M33 @ _corners(w, h)
    x0, y0 = float(tc[0].min()), float(tc[1].min())
    x1, y1 = float(tc[0].max()), float(tc[1].max())
    W, H = int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))
    T = np.eye(3, dtype=np.float32); T[0,2], T[1,2] = -x0, -y0
    return (W, H), T

def _rot_around(cx, cy, deg):
    return _to33(cv.getRotationMatrix2D((cx, cy), deg, 1.0))

def _scale_around(cx, cy, sx, sy):
    T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], np.float32)
    S  = np.array([[sx,0,0],[0,sy,0],[0,0,1]],   np.float32)
    T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]],   np.float32)
    return T2 @ (S @ T1)

def _warp(img, M33):
    h, w = img.shape[:2]
    (W, H), T = _bbox_for(M33, w, h)
    return cv.warpAffine(img, _to23(T @ M33), (W, H), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)

# --- modos ---
def _mode1(img):  # escalar x2 -> rotar 45° (2 remuestreos)
    s = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
    h, w = s.shape[:2]
    return _warp(s, _rot_around(w/2, h/2, 45))

def _mode2(img):  # (escala x2 + rotar 45°) en una sola warp
    h, w = img.shape[:2]; cx, cy = w/2, h/2
    return _warp(img, _rot_around(cx, cy, 45) @ _scale_around(cx, cy, 2, 2))

def _mode3(img):  # rotar 90° + escalar x2 en una warp
    h, w = img.shape[:2]; cx, cy = w/2, h/2
    return _warp(img, _scale_around(cx, cy, 2, 2) @ _rot_around(cx, cy, 90))

def _hud(dst, mode, video_on):
    y=24; pad=10
    for line in (
        f"p05 bilineal | modo:{mode} | fuente:{'VIDEO' if video_on else 'IMAGEN'}",
        "[1] x2->rot45 (2 warps)  [2] (x2+rot45) 1 warp  [3] rot90+x2 1 warp",
        "[v] video/img  [s] snapshot  [o] abrir img  [q]/ESC salir",
    ):
        cv.putText(dst, line, (pad,y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv.LINE_AA)
        cv.putText(dst, line, (pad,y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv.LINE_AA)
        y += 22

def _ensure_3c(img):
    if img is None: return None
    if img.ndim == 2: return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    if img.shape[2] == 1: return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img

# --- run ---
@register("p05", "Transformaciones bilineales (3 modos)")
def run(device="/dev/video2", width=640, height=480, fps=30):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    win = "p05_bilineal"

    # cámara
    try:
        if _USE_CAMKIT:
            cap, idx = open_camera(preferred_index=2, prefer_label="DroidCam")
            stream = frame_stream(cap, idx, auto_recover=True, max_read_fails=20, verbose=False)
        else:
            raise RuntimeError("CamKit no disponible")
    except Exception as e:
        print(f"[p05] fallback cámara: {e}")
        cap, stream = _fallback_open(device=device, w=width, h=height, fps=fps)

    cv.namedWindow(win, cv.WINDOW_NORMAL)
    mode, video_on, image_src = 1, True, None

    try:
        for ok, frame in stream:
            if not ok or frame is None:
                cv.waitKey(1); continue
            frame = _ensure_3c(frame)

            src = frame if video_on else (image_src if image_src is not None else frame)
            if   mode == 1: out = _mode1(src)
            elif mode == 2: out = _mode2(src)
            else:           out = _mode3(src)

            out = _ensure_3c(out); _hud(out, mode, video_on)
            cv.imshow(win, out)

            k = cv.waitKey(1) & 0xFF
            if k in (27, ord('q')): break
            elif k == ord('1'): mode = 1
            elif k == ord('2'): mode = 2
            elif k == ord('3'): mode = 3
            elif k == ord('v'): video_on = not video_on
            elif k == ord('s'): image_src, video_on = frame.copy(), False
            elif k == ord('o'):
                try:
                    print("\nRuta de imagen (ENTER cancela): ", end="", flush=True)
                    path = input().strip()
                    if path:
                        img = cv.imread(path, cv.IMREAD_COLOR)
                        if img is not None:
                            image_src, video_on = img, False
                            print(f"[OK] {path} {img.shape[1]}x{img.shape[0]}")
                        else:
                            print("[ERR] No pude leer la imagen.")
                    else:
                        print("[!] Cancelado.")
                except Exception as e:
                    print(f"[ERR] input(): {e}")
    finally:
        try: cap.release()
        except Exception: pass
        cv.destroyAllWindows()

if __name__ == "__main__":
    run()
