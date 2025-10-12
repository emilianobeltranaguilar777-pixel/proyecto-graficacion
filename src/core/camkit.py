import time
import subprocess
import cv2 as cv

# ---------------------------------------------------------
# CamKit: abrir DroidCam estable y dar frames con autorecuperación
# Preferencia: GStreamer (funciona en tu equipo) -> V4L2 + I420
# ---------------------------------------------------------

def _warmup(cap, n=8):
    ok_any = False
    for _ in range(n):
        ok, _frm = cap.read()
        cv.waitKey(1)
        if ok:
            ok_any = True
        else:
            time.sleep(0.02)
    return ok_any

def _set_i420_props(cap, w=640, h=480, fps=30):
    try:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"I420"))
    except Exception:
        pass
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv.CAP_PROP_FPS,          fps)

import os

def _restart_feeder():
    """Reinicia el feeder solo si el usuario lo permite explícitamente.
    Evitamos bloquear la GUI pidiendo contraseña sudo.
    Habilita con: export CAMKIT_ALLOW_SUDO=1
    """
    if os.environ.get("CAMKIT_ALLOW_SUDO")=="1":
        try:
            subprocess.run(
                ["/usr/bin/sudo","-n","systemctl","restart","droidcam.service"],
                check=False, timeout=1.5
            )
            time.sleep(0.5)
        except Exception:
            pass
    # Si no está habilitado, no hacemos nada (evitamos bloqueo).

def open_camera(preferred_index=2, prefer_label="DroidCam"):
    """
    Intenta abrir DroidCam en /dev/video{preferred_index}.
    1) GStreamer pipeline (te funcionó con gst-launch).
    2) V4L2 con FOURCC I420 + warm-up.
    Reinicia el feeder si la primera lectura falla.
    """
    idx = preferred_index if preferred_index is not None else 2

    # ----- Intento 1: GStreamer -----
    gst = (
        f"v4l2src device=/dev/video{idx} ! "
        "video/x-raw,format=I420,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! appsink drop=true sync=false"
    )
    cap = cv.VideoCapture(gst, cv.CAP_GSTREAMER)
    if cap.isOpened() and _warmup(cap):
        return cap, idx
    if cap:
        cap.release()

    # ----- Intento 2: V4L2 + I420 -----
    cap = cv.VideoCapture(idx, cv.CAP_V4L2)
    _set_i420_props(cap, 640, 480, 30)
    if cap.isOpened() and _warmup(cap):
        return cap, idx
    if cap:
        cap.release()

    # ----- Intento 3: restart feeder + V4L2 -----
    _restart_feeder()
    cap = cv.VideoCapture(idx, cv.CAP_V4L2)
    _set_i420_props(cap, 640, 480, 30)
    if cap.isOpened() and _warmup(cap):
        return cap, idx

    # ----- Intento 4: restart feeder + GStreamer -----
    cap = cv.VideoCapture(gst, cv.CAP_GSTREAMER)
    if cap.isOpened() and _warmup(cap):
        return cap, idx

    raise RuntimeError(f"No se pudo abrir /dev/video{idx} (GStreamer ni V4L2-I420)")

def frame_stream(cap, idx, auto_recover=True, max_read_fails=20, verbose=False):
    """Generador de frames con autorecuperación."""
    fails = 0
    while True:
        ok, frame = cap.read()
        if ok and frame is not None:
            fails = 0
            yield True, frame
            continue

        fails += 1
        if not auto_recover or fails < max_read_fails:
            yield False, None
            cv.waitKey(1)
            continue

        # Recuperación
        if verbose:
            print("[CamKit] Recuperando la cámara…")
        try:
            cap.release()
        except Exception:
            pass
        # Reabrir usando la misma lógica que open_camera
        try:
            new_cap, _ = open_camera(preferred_index=idx)
            cap = new_cap
            fails = 0
            yield False, None
        except Exception as e:
            if verbose:
                print(f"[CamKit] Recuperación falló: {e}")
            time.sleep(0.2)
            yield False, None
