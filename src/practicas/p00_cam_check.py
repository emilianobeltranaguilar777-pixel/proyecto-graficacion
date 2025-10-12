"""
p00_cam_check.py
Preview mínimo para validar el feed de DroidCam con CamKit.
Teclas:
  [q] salir
  [i] imprimir {width}x{height} @ fps (propiedad + estimado reciente)
Ejecución (Wayland/Hyprland): export QT_QPA_PLATFORM=xcb
"""

import os
import sys
import time

# Para Wayland/Hyprland, preferimos xcb con OpenCV HighGUI
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# Asegurar que 'src/' está en PYTHONPATH al ejecutar como script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import cv2 as cv  # noqa: E402
from src.core.camkit import open_camera, frame_stream  # noqa: E402


def main():
    # Abrir cámara (preferimos /dev/video2 por convenio del proyecto)
    cap, idx = open_camera(preferred_index=2, prefer_label="DroidCam")
    win = "Preview - CamKit"
    cv.namedWindow(win, cv.WINDOW_AUTOSIZE)

    # Medición simple de FPS observado
    fps_prop = cap.get(cv.CAP_PROP_FPS) or 0.0
    obs_count = 0
    obs_start = time.time()
    obs_fps = 0.0

    for ok, frame in frame_stream(cap, idx, auto_recover=True, max_read_fails=20, verbose=True):
        if ok and frame is not None:
            obs_count += 1
            now = time.time()
            dt = now - obs_start
            if dt >= 2.0:
                obs_fps = obs_count / dt
                obs_count = 0
                obs_start = now

            cv.imshow(win, frame)
        else:
            # Cuando no hay frame, aún atendemos eventos de GUI
            cv.waitKey(1)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('i'):
            h, w = (frame.shape[0], frame.shape[1]) if ok and frame is not None else (-1, -1)
            print(f"[INFO] {w}x{h} @ CAP_FPS={fps_prop:.2f}, OBS_FPS~{obs_fps:.2f}")

    try:
        cap.release()
    except Exception:
        pass
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
