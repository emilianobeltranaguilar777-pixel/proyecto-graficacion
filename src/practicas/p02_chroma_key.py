
import os
import cv2 as cv
import numpy as np

__title__ = "Capa de INVISIVILIDAD"

# --- Registro de la Practica --------------------------
try:
    from src.core.registry import register
except Exception:
    try:
        from core.registry import register
    except Exception:
        def register(*args, **kwargs):
            def deco(fn): return fn
            return deco

# --- CamKit (para abir la camara del cel) + fallback --------------------------------
_USE_CAMKIT = False
try:
    from src.core.camkit import open_camera, frame_stream  # igual que p01
    _USE_CAMKIT = True
except Exception:
    try:
        from core.camkit import open_camera, frame_stream
        _USE_CAMKIT = True
    except Exception:
        _USE_CAMKIT = False

def _fallback_open_camera(dev="/dev/video2", width=640, height=480, fps=30):
    cap = cv.VideoCapture(dev, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS,          fps)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir la cámara: {dev}")
    return cap

def _fallback_frame_stream(cap):
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield ok, frame  # (ok, frame) igual que frame_stream de CamKit

# --- Presets HSV (COLORES PARA DISTINTOS CHROMAS) ---------------------------------------------------
COLOR_PRESETS = {
    "verde":    (np.array([35,  60,  40], np.uint8), np.array([85, 255, 255], np.uint8)),
    "azul":     (np.array([90,  60,  40], np.uint8), np.array([130,255, 255], np.uint8)),
    "rojo_a":   (np.array([0,   80,  50], np.uint8), np.array([10, 255, 255], np.uint8)),
    "rojo_b":   (np.array([170, 80,  50], np.uint8), np.array([179,255, 255], np.uint8)),
    "blanco":   (np.array([0,   0,  200], np.uint8), np.array([179, 40, 255], np.uint8)),
    "amarillo": (np.array([20,  80,  60], np.uint8), np.array([35, 255, 255], np.uint8)),
    "magenta":  (np.array([140, 80,  50], np.uint8), np.array([169,255, 255], np.uint8)),
}
PRESET_KEYS = {
    ord('1'): "verde",
    ord('2'): "azul",
    ord('3'): "rojo",     # aja sabemos que el rojo son dos secciones en hsv por eso los combinamos
    ord('4'): "blanco",
    ord('5'): "amarillo",
    ord('6'): "magenta",
}

def _mask_from_preset(hsv, name):
    if name == "rojo":
        lo1, hi1 = COLOR_PRESETS["rojo_a"]
        lo2, hi2 = COLOR_PRESETS["rojo_b"]
        return cv.inRange(hsv, lo1, hi1) | cv.inRange(hsv, lo2, hi2)
    lo, hi = COLOR_PRESETS[name]
    return cv.inRange(hsv, lo, hi)

# --- Práctica ---------------------------------------------------------------
@register("p02", "Capa de INVISIVILIDAD")
def run(device="/dev/video2", width=640, height=480, fps=30, warmup_ms=800):
    # Wayland tip
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    # Abre camara
    try:
        if _USE_CAMKIT:
            cap, idx = open_camera(preferred_index=2, prefer_label="DroidCam")
            stream = frame_stream(cap, idx, auto_recover=True, max_read_fails=20, verbose=False)
        else:
            raise RuntimeError("CamKit no disponible")
    except Exception as e:
        print(f"[p02] CamKit no disponible ({e}); usando OpenCV fallback")
        cap = _fallback_open_camera(device, width, height, fps)
        stream = _fallback_frame_stream(cap)

    cv.namedWindow("p02_chroma_simple", cv.WINDOW_NORMAL)
    if warmup_ms > 0:
        cv.waitKey(warmup_ms)

    bg = None
    preset = "verde"

    # Capturar primer fondo válido
    for _ in range(15):
        ok, frame = next(stream, (False, None))
        if not ok or frame is None:
            continue
        bg = frame.copy()
        break
    if bg is None:
        print("[p02] No pude capturar fondo inicial.")
        try:
            cap.release()
        except Exception:
            pass
        cv.destroyAllWindows()
        return

    # Bucle principal
    for ok, frame in stream:
        if not ok or frame is None:
            cv.waitKey(1)
            continue

        # tamaño del fondo
        if bg.shape[:2] != frame.shape[:2]:
            bg = cv.resize(bg, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_LINEAR)

        #mascaraa
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = _mask_from_preset(hsv, preset)
        mask_inv = cv.bitwise_not(mask)
        keep_fg = cv.bitwise_and(frame, frame, mask=mask_inv)
        put_bg  = cv.bitwise_and(bg,    bg,    mask=mask)
        out = cv.add(keep_fg, put_bg)

        # UI mínima
        ui = out
        cv.putText(ui, f"[p02 simple] Presets 1:Verde 2:Azul 3:Rojo 4:Blanco 5:Amarillo 6:Magenta | c:Capturar fondo | q:Salir",
                   (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(ui, f"[p02 simple] Presets 1:Verde 2:Azul 3:Rojo 4:Blanco 5:Amarillo 6:Magenta | c:Capturar fondo | q:Salir",
                   (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv.LINE_AA)

        cv.imshow("p02_capa_de_invisilivilidad", ui)

        k = cv.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        if k in PRESET_KEYS:
            preset_name = PRESET_KEYS[k]
            preset = preset_name
        if k == ord('c'):
            bg = frame.copy()
            print("[p02] Nuevo fondo capturado.")

    # Cleanup
    try:
        cap.release()
    except Exception:
        pass
    cv.destroyAllWindows()

# --- Ejecutable directo -----------------------------------------------------
def _parse_argv():
    import argparse
    p = argparse.ArgumentParser(description="p02 - Chroma key en vivo (simple)")
    p.add_argument("--device", default="/dev/video2", help="Ruta de cámara")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=800)
    return p.parse_args()

if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    args = _parse_argv()
    run(device=args.device, width=args.width, height=args.height, fps=args.fps, warmup_ms=args.warmup)
