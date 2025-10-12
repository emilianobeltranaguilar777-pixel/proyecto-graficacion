"""
P01 – Filtro Mosaico (DroidCam)
--------------------------------
Objetivo:
  - Tomar video en vivo desde la cámara del móvil (DroidCam en /dev/video2).
  - Construir un mosaico (2x3) con variaciones del mismo frame.
  - Ofrecer tres modos:
      [1] RGB: separación/mezcla de canales (R,G,B y combinaciones).
      [2] Blanco/Negro: variaciones en escala de grises (inverso, binarización, etc.).
      [3] 5 Colores: cuantización del frame a 5 colores (paleta aleatoria).

Controles:
  - [1]  -> Modo RGB
  - [2]  -> Modo Blanco/Negro
  - [3]  -> Modo 5 Colores
  - [m]  -> Ciclar siguiente modo
  - [r]  -> Re-rodar paleta (solo en modo 5 Colores)
  - [s]  -> Guardar snapshot del mosaico (carpeta ./assets si existe, si no en ./)
  - [i]  -> Imprimir tamaño de frame y fps estimado
  - [q] o [Esc] -> Salir

Notas:
  - Se apoya en core.camkit.open_camera/frame_stream para robustez (autorreconexión).
  - En Wayland/Hyprland exporta antes de ejecutar:  QT_QPA_PLATFORM=xcb
"""

import os
import time
import random
import string
import numpy as np
import cv2 as cv

from src.core.registry import register
from src.core.camkit import open_camera, frame_stream


# ---------------------------- Utilidades de imagen ----------------------------

def _bgr_only(frame, which="r"):
    """Devuelve frame manteniendo solo un canal (R,G,B) visible."""
    b, g, r = cv.split(frame)
    zeros = np.zeros_like(b)
    if which.lower() == "r":
        return cv.merge([zeros, zeros, r])
    if which.lower() == "g":
        return cv.merge([zeros, g, zeros])
    return cv.merge([b, zeros, zeros])  # "b" por default


def _bgr_combo(frame, keep=("r", "g")):
    """Devuelve combinaciones de canales (e.g., RG, RB, GB)."""
    b, g, r = cv.split(frame)
    zeros = np.zeros_like(b)
    m = {
        ("r", "g"): cv.merge([zeros, g, r]),
        ("r", "b"): cv.merge([b, zeros, r]),
        ("g", "b"): cv.merge([b, g, zeros]),
    }
    key = tuple(sorted([c.lower() for c in keep]))
    return m.get(key, frame)


def _to_gray(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def _to_gray_bgr(frame):
    g = _to_gray(frame)
    return cv.cvtColor(g, cv.COLOR_GRAY2BGR)


def _invert_gray(frame_gray):
    return 255 - frame_gray


def _binary(frame_gray, thresh=127):
    _, bin_img = cv.threshold(frame_gray, thresh, 255, cv.THRESH_BINARY)
    return bin_img


def _adaptive_binary(frame_gray):
    return cv.adaptiveThreshold(frame_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, 31, 5)


def _clahe_gray(frame_gray):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(frame_gray)


def _edges_bgr(frame):
    g = _to_gray(frame)
    e = cv.Canny(g, 80, 160)
    return cv.cvtColor(e, cv.COLOR_GRAY2BGR)


def _random_palette(k=5, seed=None):
    """Genera paleta aleatoria de k colores BGR (valores 0..255)."""
    if seed is not None:
        rnd = random.Random(seed)
        return np.array([[rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255)]
                         for _ in range(k)], dtype=np.uint8)
    return np.random.randint(0, 256, size=(k, 3), dtype=np.uint8)


def _quantize_to_palette(frame_bgr, palette_bgr, downscale=4):
    """
    Cuantiza el frame a la paleta dada por distancia euclidiana en BGR.
    Para rendimiento: opera en imagen reducida y reescala al final.
    """
    h, w = frame_bgr.shape[:2]
    ds = max(1, int(downscale))
    small = cv.resize(frame_bgr, (w // ds, h // ds), interpolation=cv.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.int16)  # (N,3)
    pal = palette_bgr.astype(np.int16)             # (K,3)

    # Distancias (N,K): ||pixel - color||^2
    # Expand dims para broadcasting: (N,1,3) - (1,K,3) => (N,K,3) -> sum eje 2
    dists = np.sum((pixels[:, None, :] - pal[None, :, :]) ** 2, axis=2)  # (N,K)
    labels = np.argmin(dists, axis=1)                                     # (N,)
    quant_small = pal[labels].astype(np.uint8).reshape(small.shape)
    return cv.resize(quant_small, (w, h), interpolation=cv.INTER_NEAREST)


def _palette_swatch(palette_bgr, tile_size=(240, 180)):
    """Dibuja un mosaico pequeño que muestra los 5 colores de la paleta."""
    w, h = tile_size
    k = palette_bgr.shape[0]
    sw = np.zeros((h, w, 3), dtype=np.uint8)
    cw = w // k
    for i, color in enumerate(palette_bgr):
        sw[:, i * cw:(i + 1) * cw] = color[None, None, :]
    # Borde
    cv.rectangle(sw, (0, 0), (w - 1, h - 1), (255, 255, 255), 2)
    return sw


def _put_label(img, text, org=(8, 20)):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv.LINE_AA)
    return img


def _make_mosaic(tiles, grid=(2, 3)):
    """Compone lista de tiles (BGR) en un mosaico grid=(rows,cols). Ajusta tamaños automáticamente."""
    rows, cols = grid
    assert len(tiles) == rows * cols, f"Se esperaban {rows*cols} tiles y llegaron {len(tiles)}"
    # Tamaño base = tamaño del primer tile
    th, tw = tiles[0].shape[:2]
    # Asegurar todos del mismo tamaño visual
    tiles_resized = [cv.resize(t, (tw, th), interpolation=cv.INTER_AREA) for t in tiles]
    # Apilar
    bands = []
    for r in range(rows):
        row_tiles = tiles_resized[r * cols:(r + 1) * cols]
        bands.append(np.hstack(row_tiles))
    return np.vstack(bands)


# ---------------------------- Lógica de modos ----------------------------

MODE_RGB = 0
MODE_BN = 1
MODE_5C = 2
MODE_NAMES = {MODE_RGB: "RGB", MODE_BN: "Blanco/Negro", MODE_5C: "5 Colores"}

def _tiles_rgb(frame):
    """Tiles para modo RGB: original, R, G, B, RG, GB."""
    t0 = frame
    t1 = _bgr_only(frame, "r")
    t2 = _bgr_only(frame, "g")
    t3 = _bgr_only(frame, "b")
    t4 = _bgr_combo(frame, ("r", "g"))
    t5 = _bgr_combo(frame, ("g", "b"))
    # Etiquetas
    _put_label(t0, "Original")
    _put_label(t1, "Solo R")
    _put_label(t2, "Solo G")
    _put_label(t3, "Solo B")
    _put_label(t4, "R+G")
    _put_label(t5, "G+B")
    return [t0, t1, t2, t3, t4, t5]


def _tiles_bn(frame):
    """Tiles para modo B/N: Gray, Invert, CLAHE, Binary, Adaptive, Edges."""
    g = _to_gray(frame)
    t0 = _to_gray_bgr(frame)                     # Gray
    t1 = cv.cvtColor(_invert_gray(g), cv.COLOR_GRAY2BGR)  # Invert
    t2 = cv.cvtColor(_clahe_gray(g), cv.COLOR_GRAY2BGR)   # CLAHE
    t3 = cv.cvtColor(_binary(g, 127), cv.COLOR_GRAY2BGR)  # Binary
    t4 = cv.cvtColor(_adaptive_binary(g), cv.COLOR_GRAY2BGR)  # Adaptive
    t5 = _edges_bgr(frame)                        # Edges
    # Etiquetas
    _put_label(t0, "Gris")
    _put_label(t1, "Inverso")
    _put_label(t2, "CLAHE")
    _put_label(t3, "Binaria")
    _put_label(t4, "Adaptativa")
    _put_label(t5, "Bordes")
    return [t0, t1, t2, t3, t4, t5]


def _tiles_5c(frame, palette):
    """Tiles para modo 5 Colores: Original, 5-colors, + swatch y variaciones."""
    q = _quantize_to_palette(frame, palette)
    # Variaciones simples sobre cuantizada
    q_inv = 255 - q
    q_blur = cv.GaussianBlur(q, (7, 7), 0)
    sw = _palette_swatch(palette, tile_size=(frame.shape[1] // 3, frame.shape[0] // 3))
    sw = cv.resize(sw, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_NEAREST)

    t0 = frame.copy()
    t1 = q
    t2 = q_inv
    t3 = q_blur
    t4 = sw
    t5 = _edges_bgr(q)

    _put_label(t0, "Original")
    _put_label(t1, "5 Colores")
    _put_label(t2, "5 Colores (invert)")
    _put_label(t3, "5 Colores (blur)")
    _put_label(t4, "Paleta")
    _put_label(t5, "Bordes 5C")
    return [t0, t1, t2, t3, t4, t5]


# ---------------------------- Práctica registrada ----------------------------

@register("p01", "Filtro Mosaico (RGB / B&N / 5 Colores) - DroidCam")
def run():
    # Abrir cámara preferentemente en /dev/video2 (DroidCam). CamKit maneja formato I420 y warm-up.
    cap, idx = open_camera(preferred_index=2, prefer_label="DroidCam")

    win = "P01 - Filtro Mosaico (DroidCam)"
    cv.namedWindow(win, cv.WINDOW_AUTOSIZE)

    mode = MODE_RGB
    palette = _random_palette(5)  # Paleta inicial para modo 5C
    last_t = time.time()
    frames = 0
    fps_est = 0.0

    for ok, frame in frame_stream(cap, idx, auto_recover=True, max_read_fails=20, verbose=False):
        if not ok or frame is None:
            cv.waitKey(1)
            continue

        # FPS estimado (suave)
        frames += 1
        if frames % 15 == 0:
            now = time.time()
            dt = max(1e-6, now - last_t)
            fps_est = 15.0 / dt
            last_t = now

        # Construcción de tiles según modo
        if mode == MODE_RGB:
            tiles = _tiles_rgb(frame.copy())
        elif mode == MODE_BN:
            tiles = _tiles_bn(frame.copy())
        else:
            tiles = _tiles_5c(frame.copy(), palette)

        mosaic = _make_mosaic(tiles, grid=(2, 3))

        # HUD con estado y ayuda rápida
        hud = f"Modo: {MODE_NAMES[mode]} | FPS~ {fps_est:.1f} | [1]RGB [2]B/N [3]5C  [m]siguiente  [r]paleta  [s]snap  [q]salir"
        _put_label(mosaic, hud, org=(8, 28))

        cv.imshow(win, mosaic)
        k = cv.waitKey(1) & 0xFF

        if k in (27, ord('q')):
            break
        elif k == ord('m'):
            mode = (mode + 1) % 3
        elif k == ord('1'):
            mode = MODE_RGB
        elif k == ord('2'):
            mode = MODE_BN
        elif k == ord('3'):
            mode = MODE_5C
        elif k == ord('r'):
            if mode == MODE_5C:
                palette = _random_palette(5)
        elif k == ord('i'):
            h, w = frame.shape[:2]
            print(f"[P01] Frame: {w}x{h}  | FPS estimado: {fps_est:.2f}")
        elif k == ord('s'):
            # Guardar snapshot del mosaico
            ts = time.strftime("%Y%m%d-%H%M%S")
            rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
            base_dir = "./assets" if os.path.isdir("./assets") else "."
            path = os.path.join(base_dir, f"p01_mosaico_{MODE_NAMES[mode]}_{ts}_{rand}.png")
            cv.imwrite(path, mosaic)
            print(f"[P01] Snapshot guardado en: {path}")

    cap.release()
    cv.destroyAllWindows()
