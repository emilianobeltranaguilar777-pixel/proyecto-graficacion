#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p06 — Aprendiendo a usar MediaPipe
Controles: [F] cara · [H] manos · [B] cuerpo · [S] guardar captura · [Q]/ESC salir
"""
import os, time
import cv2 as cv
import numpy as np
from datetime import datetime

# --- registro (2 args, sin tag) ---
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
            if not ok: yield False, None
            else:      yield True, f
    return cap, stream()

def _hud(img, face, hands, body, fps):
    y=24; pad=10
    box = img.copy()
    cv.rectangle(box, (pad-6, 6), (340, 120), (0,0,0), -1)
    cv.addWeighted(box, 0.45, img, 0.55, 0, img)
    lines = [
        "p05 — Aprendiendo MediaPipe",
        f"[F] Cara:   {'ON' if face  else 'OFF'}",
        f"[H] Manos:  {'ON' if hands else 'OFF'}",
        f"[B] Cuerpo: {'ON' if body  else 'OFF'}",
        f"FPS: {fps:0.1f}" if fps is not None else ""
    ]
    for line in lines:
        if not line: continue
        cv.putText(img, line, (pad, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv.LINE_AA)
        cv.putText(img, line, (pad, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv.LINE_AA)
        y += 22

@register("p06", "Aprendiendo a usar MediaPipe")
def run():
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    # Importar mediapipe dentro de run() para fallar bonito si falta
    try:
        import mediapipe as mp
        mp_draw   = mp.solutions.drawing_utils
        mp_style  = mp.solutions.drawing_styles
        mp_hol    = mp.solutions.holistic
    except Exception as e:
        canvas = np.zeros((280, 720, 3), np.uint8)
        msg = ("No se pudo importar mediapipe.\n"
               "Instala en tu venv:\n"
               "  pip install --upgrade pip\n"
               "  pip install mediapipe\n\n"
               f"Detalle: {type(e).__name__}: {e}")
        for i, line in enumerate(msg.splitlines()):
            cv.putText(canvas, line, (20, 40+28*i), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)
        cv.namedWindow("p05_mediapipe", cv.WINDOW_NORMAL)
        cv.imshow("p05_mediapipe", canvas)
        while True:
            if (cv.waitKey(0) & 0xFF) in (27, ord('q')): break
        cv.destroyAllWindows()
        return

    # Cámara
    try:
        if _USE_CAMKIT:
            cap, idx = open_camera(preferred_index=2, prefer_label="DroidCam")
            stream = frame_stream(cap, idx, auto_recover=True, max_read_fails=20, verbose=False)
        else:
            raise RuntimeError("CamKit no disponible")
    except Exception as e:
        print(f"[p05] fallback cámara: {e}")
        cap, stream = _fallback_open()

    cv.namedWindow("p06_mediapipe", cv.WINDOW_NORMAL)
    show_face, show_hands, show_body = True, True, True
    t0, frames, fps = time.time(), 0, None

    hol_kwargs = dict(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        with mp_hol.Holistic(**hol_kwargs) as hol:
            for ok, bgr in stream:
                if not ok or bgr is None:
                    cv.waitKey(1); continue

                # FPS simple
                frames += 1
                if frames % 10 == 0:
                    dt = time.time() - t0
                    fps = 10.0/dt if dt > 0 else None
                    t0 = time.time()

                # MediaPipe procesa en RGB
                rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = hol.process(rgb)
                out = bgr

                if show_body and res.pose_landmarks:
                    mp_draw.draw_landmarks(
                        out, res.pose_landmarks, mp_hol.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
                    )
                if show_hands:
                    if res.left_hand_landmarks:
                        mp_draw.draw_landmarks(
                            out, res.left_hand_landmarks, mp_hol.HAND_CONNECTIONS,
                            mp_style.get_default_hand_landmarks_style(),
                            mp_style.get_default_hand_connections_style()
                        )
                    if res.right_hand_landmarks:
                        mp_draw.draw_landmarks(
                            out, res.right_hand_landmarks, mp_hol.HAND_CONNECTIONS,
                            mp_style.get_default_hand_landmarks_style(),
                            mp_style.get_default_hand_connections_style()
                        )
                if show_face and res.face_landmarks:
                    mp_draw.draw_landmarks(
                        out, res.face_landmarks, mp_hol.FACEMESH_TESSELATION,
                        None, mp_style.get_default_face_mesh_tesselation_style()
                    )
                    mp_draw.draw_landmarks(
                        out, res.face_landmarks, mp_hol.FACEMESH_CONTOURS,
                        None, mp_style.get_default_face_mesh_contours_style()
                    )
                    mp_draw.draw_landmarks(
                        out, res.face_landmarks, mp_hol.FACEMESH_IRISES,
                        None, mp_style.get_default_face_mesh_iris_connections_style()
                    )

                _hud(out, show_face, show_hands, show_body, fps)
                cv.imshow("p05_mediapipe", out)

                k = cv.waitKey(1) & 0xFF
                if k in (27, ord('q')): break
                elif k in (ord('f'), ord('F')): show_face  = not show_face
                elif k in (ord('h'), ord('H')): show_hands = not show_hands
                elif k in (ord('b'), ord('B')): show_body  = not show_body
                elif k in (ord('s'), ord('S')):
                    os.makedirs("capturas", exist_ok=True)
                    path = f"capturas/p05_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv.imwrite(path, out)
                    cv.displayOverlay("p06_mediapipe", f"Guardado: {path}", 1500)
    finally:
        try: cap.release()
        except Exception: pass
        cv.destroyAllWindows()

if __name__ == "__main__":
    run()
