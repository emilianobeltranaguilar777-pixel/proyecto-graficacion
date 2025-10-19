#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 p03 — ASCII en vivo (video → texto)

 Convierte el stream de la cámara en una rejilla de caracteres ASCII renderizados
 en un lienzo OpenCV. Usa CamKit (open_camera/frame_stream) y hace fallback a
 /dev/video2 si CamKit falla.

 Controles:
   q / ESC  → salir
   +/-      → aumentar / disminuir densidad (tamaño de celda)
   g        → ASCII coloreado por bloque (media de color) ON/OFF
   i        → invertir la rampa de caracteres (oscuro↔claro)
   s        → guardar snapshot en ./assets/ (si existe) o en cwd

"""

__title__ = "ASCII en vivo"

from src.core.registry import register
from src.core.camkit import open_camera, frame_stream

import os
import time
import cv2 as cv
import numpy as np
from pathlib import Path


# ------------------------ Utilidades internas  ------------------------
_ASCII_RAMP = np.array(list(" .'`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"))
# Nota: anteriores contienen barras y comillas escapadas.


def _build_ascii_lut(invert: bool = False):
    ramp = _ASCII_RAMP[::-1] if invert else _ASCII_RAMP
    # Creamos LUT de 0..255 → índice en rampa
    # Usamos 256 valores mapeados a len(ramp)
    idx = (np.linspace(0, len(ramp) - 1, 256)).astype(np.uint8)
    return ramp, idx


def _text_canvas(cols: int, rows: int, cell_w: int, cell_h: int, colored: bool) -> np.ndarray:
    """Crea un lienzo RGB en blanco para dibujar texto en celdas (una letra por celda)."""
    h = rows * cell_h
    w = cols * cell_w
    if colored:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)  # negro para que el color contraste
    else:
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)  # blanco
    return canvas


def _put_char(img, ch: str, x: int, y: int, cell_w: int, cell_h: int, color=(0, 0, 0)):
    # Posicionamos baseline dentro de la celda con márgenes pequeños
    org = (x + max(1, cell_w // 10), y + cell_h - max(1, cell_h // 5))
    cv.putText(img, ch, org, cv.FONT_HERSHEY_PLAIN, 
               fontScale=max(0.8, min(cell_w, cell_h) / 10.0),
               color=tuple(int(c) for c in color), thickness=1, lineType=cv.LINE_AA)


# ------------------------ Fallback estándar solicitado ------------------------

def _fallback_open(device="/dev/video2", w=640, h=480, fps=30):
    cap = cv.VideoCapture(device, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir {device}")

    def stream():
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield ok, f

    return cap, stream()


# ------------------------ run() principal ------------------------

@register("p03", "ASCII en vivo")
def run():
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    # 1) Abrir cámara con CamKit, si falla usar fallback
    try:
        cap, idx = open_camera(preferred_index=2, prefer_label="DroidCam")
        stream = frame_stream(cap, idx, auto_recover=True, max_read_fails=20, verbose=False)
    except Exception:
        cap, stream = _fallback_open()

    # 2) Parámetros del efecto
    cell_w, cell_h = 8, 12   # tamaño de celda inicial (px)
    colored = False          # color por bloque
    invert = False           # invertir rampa
    ramp, lut = _build_ascii_lut(invert)

    win_name = "p03_ascii_cam"
    help_name = "p03_ascii_cam_help"
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.namedWindow(help_name, cv.WINDOW_NORMAL)

    last_fps_t = time.time()
    frames = 0
    fps_txt = ""

    # 3) Bucle principal: consumir (ok, frame)
    for ok, frame in stream:
        if not ok or frame is None:
            cv.waitKey(1)
            continue

        frames += 1
        now = time.time()
        if now - last_fps_t >= 0.5:
            fps_txt = f"{frames / (now - last_fps_t):.1f} FPS"
            last_fps_t = now
            frames = 0

        # 3.1) Preprocesado → tamaño discreto a celdas
        h, w = frame.shape[:2]
        cols = max(8, w // cell_w)
        rows = max(6, h // cell_h)

        # Para ASCII: muestreamos por bloques
        small = cv.resize(frame, (cols, rows), interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)

        # 3.2) Mapear intensidades → caracteres
        chars_idx = lut[gray]  # (rows, cols)
        chars = ramp[chars_idx]  # array de dtype '<U1'

        # 3.3) Lienzo y pintado
        canvas = _text_canvas(cols, rows, cell_w, cell_h, colored)
        if colored:
            # color = media del bloque (BGR) ya reducido
            for r in range(rows):
                y = r * cell_h
                for c in range(cols):
                    x = c * cell_w
                    ch = str(chars[r, c])
                    color = small[r, c].tolist()  # BGR
                    _put_char(canvas, ch, x, y, cell_w, cell_h, color=color)
        else:
            for r in range(rows):
                y = r * cell_h
                for c in range(cols):
                    x = c * cell_w
                    ch = str(chars[r, c])
                    _put_char(canvas, ch, x, y, cell_w, cell_h, color=(0, 0, 0))

        # 3.4) Overlay con info
        cv.rectangle(canvas, (6, 6), (350, 80), (255, 255, 255) if not colored else (0, 0, 0), -1)
        cv.putText(canvas, f"p03 ASCII — {fps_txt}", (12, 28), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 0, 0) if not colored else (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(canvas, f"{cols}x{rows} celdas  cell({cell_w}x{cell_h})",
                   (12, 56), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 0, 0) if not colored else (255, 255, 255), 1, cv.LINE_AA)

        # 3.5) Ventanas
        cv.imshow(win_name, canvas)

        # Ayuda estática
        help_img = np.full((190, 540, 3), 255, dtype=np.uint8)
        y0 = 26
        lines = [
            "Controles:",
            "  q/ESC  salir",
            "  +/-    densidad (tamaño celda)",
            "  g      color por bloque ON/OFF",
            "  i      invertir rampa",
            "  s      guardar snapshot",
        ]
        for i, line in enumerate(lines):
            cv.putText(help_img, line, (10, y0 + i * 26), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)
        cv.imshow(help_name, help_img)

        # 3.6) Teclado
        k = cv.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord('+') or k == ord('='):
            # Mayor densidad → celdas más pequeñas, manteniendo mínimos
            cell_w = max(4, cell_w - 1)
            cell_h = max(6, cell_h - 1)
        elif k == ord('-') or k == ord('_'):
            cell_w = min(32, cell_w + 1)
            cell_h = min(48, cell_h + 1)
        elif k == ord('g'):
            colored = not colored
        elif k == ord('i'):
            invert = not invert
            ramp, lut = _build_ascii_lut(invert)
        elif k == ord('s'):
            # Guardar snapshot
            outdir = Path("assets") if Path("assets").exists() else Path.cwd()
            ts = time.strftime("%Y%m%d-%H%M%S")
            out = outdir / f"p03_ascii_{ts}.png"
            try:
                cv.imwrite(str(out), canvas)
            except Exception:
                pass

    # 4) Salida limpia
    try:
        cap.release()
    except Exception:
        pass
    cv.destroyAllWindows()


# Permite: python -m src.practicas.p03_ascii_cam
if __name__ == "__main__":
    run()
