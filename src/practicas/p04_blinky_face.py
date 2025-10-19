#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p04 — DETECTANDO LA CARA
- Registro con @register("p04", "..."), lógica solo en run()
- Usa CamKit (misma firma p01) y fallback a /dev/video2
- Wayland: QT_QPA_PLATFORM=xcb
- Dibuja rostro y rasgos con rectángulos/círculos/triángulos/élipses
- Parpadeo: umbral por relación H/W de ojos → “ojos locos” por unos frames
Controles: [h] ayuda  [c] toggle ojos locos  [q/ESC] salir
"""

import os
import cv2 as cv
import numpy as np
import random

__title__ = "detectando cara"

# --- Registro -------------------------------------------
try:
    from src.core.registry import register
except Exception:
    try:
        from core.registry import register
    except Exception:
        def register(*args, **kwargs):
            def deco(fn): return fn
            return deco

# --- CamKit  + fallback ------------------------------------
_USE_CAMKIT = False
try:
    from src.core.camkit import open_camera, frame_stream
    _USE_CAMKIT = True
except Exception:
    try:
        from core.camkit import open_camera, frame_stream
        _USE_CAMKIT = True
    except Exception:
        _USE_CAMKIT = False

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
            if not ok:
                yield False, None
                break
            yield ok, f
    return cap, stream()

# --- Utilidades --------------------------------------------------------------
def _put_help(img, on=True):
    if not on: return
    y = 24
    for t in [
        "p04 — Geometric Face (sin MediaPipe)",
        "[h] ayuda   [c] ojos locos on/off   [q/ESC] salir",
        "Parpadea para activar 'ojos locos' (auto).",
    ]:
        cv.putText(img, t, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv.LINE_AA)
        cv.putText(img, t, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv.LINE_AA)
        y += 22

def _rand_color():
    return (random.randint(50,255), random.randint(50,255), random.randint(50,255))

class BlinkDetector:
    """Detección simple por relación H/W de los ojos (suavizado + cooldown)."""
    def __init__(self, thresh=0.26, smooth=0.6, cooldown_frames=5):
        self.thresh = thresh
        self.smooth = smooth
        self.cooldown = cooldown_frames
        self.s_prev = None
        self.cool = 0

    def update(self, ratios):
        if not ratios:
            return False
        r = float(sum(ratios))/len(ratios)
        if self.s_prev is None:
            self.s_prev = r
        else:
            self.s_prev = self.smooth*self.s_prev + (1-self.smooth)*r
        if self.cool > 0:
            self.cool -= 1
            return False
        if self.s_prev < self.thresh:
            self.cool = self.cooldown
            return True
        return False

# --- Render geométrico (solo OpenCV) ----------------------------------------
class GeometricFaceRenderer:
    def __init__(self):
        base = cv.data.haarcascades
        self.face_cascade = cv.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.eye_cascade  = cv.CascadeClassifier(base + "haarcascade_eye.xml")
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("No se pudieron cargar cascadas Haar de OpenCV.")
        self.blink = BlinkDetector()
        self.crazy_until = 0
        self.crazy_manual = False  # toggle con tecla 'c'

    def _draw_face_geo(self, img, x, y, w, h, eyes, crazy, mouth_open_amt):
        # Marco rostro azul
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Sombrero
        brim_h   = max(12, int(h*0.12))
        crown_h  = max(30, int(h*0.38))
        crown_w  = int(w*0.7)
        crown_x  = x + int(w*0.15)
        cv.rectangle(img, (x - int(w*0.15), y - brim_h), (x + w + int(w*0.15), y), (0,0,0), -1)
        cv.rectangle(img, (crown_x, y - brim_h - crown_h), (crown_x + crown_w, y - brim_h), (0,0,0), -1)
        band_h = max(6, int(h*0.06))
        cv.rectangle(img, (crown_x, y - brim_h - band_h), (crown_x + crown_w, y - brim_h), (180,50,50), -1)

        # Orejas
        ear_h = int(0.25 * h); ear_w = int(0.12 * w)
        ear_l = np.array([(x-ear_w, y+ear_h), (x, y+int(0.5*h)), (x-ear_w, y+h-ear_h)], dtype=np.int32)
        ear_r = np.array([(x+w+ear_w, y+ear_h), (x+w, y+int(0.5*h)), (x+w+ear_w, y+h-ear_h)], dtype=np.int32)
        cv.fillConvexPoly(img, ear_l, (200,200,255)); cv.polylines(img, [ear_l], True, (120,120,255), 2, cv.LINE_AA)
        cv.fillConvexPoly(img, ear_r, (200,200,255)); cv.polylines(img, [ear_r], True, (120,120,255), 2, cv.LINE_AA)

        # Nariz
        nose_w = int(w*0.08); nose_h = int(h*0.12)
        nose = np.array([
            (x + w//2,         y + int(h*0.42)),
            (x + w//2 - nose_w, y + int(h*0.55)),
            (x + w//2 + nose_w, y + int(h*0.55)),
        ], dtype=np.int32)
        cv.fillConvexPoly(img, nose, (0,255,255))

        # Ojos
        eye_boxes = []
        if len(eyes) >= 1:
            # mapear a coords absolutas
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_boxes.append((x+ex, y+ey, ew, eh))
        else:
            # posiciones por proporción
            ew = int(w*0.18); eh = int(h*0.12)
            eye_boxes = [
                (x + int(w*0.25) - ew//2, y + int(h*0.35) - eh//2, ew, eh),
                (x + int(w*0.75) - ew//2, y + int(h*0.35) - eh//2, ew, eh),
            ]

        for (ex, ey, ew, eh) in eye_boxes:
            if crazy:
                color = _rand_color()
                jitter = random.randint(-3,3)
                rect = (ex+jitter, ey+jitter, ew, eh)
                cv.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), color, -1)
                cv.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,0,0), 2)
                cx = rect[0] + rect[2]//2 + random.randint(-5,5)
                cy = rect[1] + rect[3]//2 + random.randint(-5,5)
                r  = max(2, int(0.12*max(ew,eh)))
                cv.circle(img, (cx,cy), r, (0,0,0), -1, cv.LINE_AA)
            else:
                # borde ojo
                cv.rectangle(img, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
                # globo blanco
                rx = max(6, int(0.45*ew)); ry = max(4, int(0.45*eh))
                cv.ellipse(img, (ex+ew//2, ey+eh//2), (rx, ry), 0, 0, 360, (255,255,255), -1)
                # pupila
                cv.circle(img, (ex+ew//2, ey+eh//2), max(3, int(min(ew,eh)*0.12)), (0,0,0), -1, cv.LINE_AA)
                # ceja (línea)
                brow_y = ey - max(2, int(0.2*eh))
                cv.line(img, (ex, brow_y), (ex+ew, brow_y), (0,255,255), 2, cv.LINE_AA)

        # Boca
        mouth_th = max(4, int(0.012 * (w+h)))
        cx = x + w//2; cy = y + int(h*0.75)
        if mouth_open_amt > 0.3:
            rx = max(10, int(w*0.18 * (1 + mouth_open_amt*0.5)))
            ry = max(5,  int(h*0.07 * mouth_open_amt))
            cv.ellipse(img, (cx, cy), (rx, ry), 0, 0, 180, (0,0,255), mouth_th)
        else:
            cv.line(img, (x+int(w*0.4), cy), (x+int(w*0.6), cy), (0,0,255), mouth_th)

        # Mentón
        cv.ellipse(img, (cx, y+h), (int(w*0.22), int(h*0.1)), 0, 200, 340, (50,50,50), 2)

    def _estimate_mouth_open(self, gray, x, y, w, h):
        # ROI boca: tercio inferior central
        my1 = y + int(h*0.65); my2 = y + int(h*0.95)
        mx1 = x + int(w*0.30); mx2 = x + int(w*0.70)
        my2 = min(my2, gray.shape[0]-1); mx2 = min(mx2, gray.shape[1]-1)
        if my1 >= my2 or mx1 >= mx2: return 0.0
        roi = gray[my1:my2, mx1:mx2]
        # umbral inverso: oscuro = posible boca abierta
        _, th = cv.threshold(roi, 50, 255, cv.THRESH_BINARY_INV)
        dark = cv.countNonZero(th); tot = roi.size if roi.size>0 else 1
        ratio = dark / tot
        # integrador suave a [0..1]
        return max(0.0, min(1.0, (ratio - 0.18) / 0.5))  # tunable

    def process(self, frame_bgr, frame_idx):
        img = frame_bgr
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(100,100))
        if len(faces) == 0:
            return img

        # 1 rostro (el más grande)
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        (x, y, w, h) = faces[0]

        # detectar ojos en ROI cara
        roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20,20))
        eyes = sorted(eyes, key=lambda e: e[0])[:2]

        # razón H/W para blink
        ratios = [eh / (ew + 1e-6) for (_,_,ew,eh) in eyes]
        if self.blink.update(ratios):
            self.crazy_until = frame_idx + 15

        mouth_open_amt = self._estimate_mouth_open(gray, x, y, w, h)
        crazy = (frame_idx < self.crazy_until) or self.crazy_manual
        self._draw_face_geo(img, x, y, w, h, eyes, crazy, mouth_open_amt)
        return img

# --- run() ------------------------------------------------------------------
@register("p04", "Geometric Face (sin MediaPipe)")
def run(device="/dev/video2", width=640, height=480, fps=30):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    win = "p04_geometric_face"

    # Abrir cámara (CamKit → fallback)
    try:
        if _USE_CAMKIT:
            cap, idx = open_camera(preferred_index=2, prefer_label="DroidCam")
            stream = frame_stream(cap, idx, auto_recover=True, max_read_fails=20, verbose=False)
        else:
            raise RuntimeError("CamKit no disponible")
    except Exception as e:
        print(f"[p04] CamKit no disponible ({e}); usando OpenCV fallback")
        cap, stream = _fallback_open(device=device, w=width, h=height, fps=fps)

    cv.namedWindow(win, cv.WINDOW_NORMAL)
    renderer = GeometricFaceRenderer()
    show_help = True
    frame_idx = 0

    try:
        for ok, frame in stream:
            if not ok or frame is None:
                cv.waitKey(1)
                continue

            out = renderer.process(frame.copy(), frame_idx)
            _put_help(out, on=show_help)
            cv.imshow(win, out)

            k = cv.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k in (ord('h'), ord('H')):
                show_help = not show_help
            elif k in (ord('c'), ord('C')):
                renderer.crazy_manual = not renderer.crazy_manual

            frame_idx += 1
    finally:
        try:
            cap.release()
        except Exception:
            pass
        cv.destroyAllWindows()

# Ejecutable directo
if __name__ == "__main__":
    run()
