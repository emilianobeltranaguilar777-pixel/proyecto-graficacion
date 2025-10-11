"""Práctica 01 — Filtro Mosaico + Split/Merge con OpenCV."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np


# url de cámara móvil tipo IP Webcam; suele ser http://<ip>:8080/video
# si no prende, intentar con /video?dummy=param.mjpg o /shot.jpg en modo still
MOBILE_STREAM_URL = "http://<IP_DEL_MOVIL>:8080/video"

DEFAULT_IMAGE_PATH = "1a.png"
WINDOW_TRACKBARS = "P01 Mosaico"
WINDOW_ORIGINAL = "original"
WINDOW_CHANNELS = "canales"
WINDOW_RESULT = "resultado"

ChannelTuple = Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class CaptureState:
    """Guarda info de la fuente para reconectar sin dramas."""

    source: Union[int, str]
    cap: Optional[cv2.VideoCapture] = None
    retries_left: int = 0

    def ensure_open(self) -> Optional[cv2.VideoCapture]:
        """Checa si el capture sigue vivo; si no, intenta abrirlo."""

        if self.cap is not None and self.cap.isOpened():
            return self.cap
        if self.cap is not None:
            self.cap.release()
        self.cap = open_capture(self.source)
        return self.cap


def load_image(path: str) -> np.ndarray:
    """Carga imagen o arma gradiente fallback para que siempre jale."""

    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    # si no traes 1a.png, no hay falla: armamos un gradiente para que esto corra sí o sí
    h, w = 480, 640
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(255, 0, h, dtype=np.uint8)
    gradient = np.stack(np.meshgrid(x, y), axis=-1)
    b = gradient[..., 0]
    g = gradient[..., 1]
    r = cv2.addWeighted(b, 0.5, g, 0.5, 0)
    return cv2.merge((b, g, r))


def open_capture(source: Union[int, str]) -> Optional[cv2.VideoCapture]:
    """Abre webcam local o stream HTTP; regresa None si no jalo."""

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        print(f"no se pudo abrir la fuente {source}, seguimos con fallback")
        return None
    return cap


def read_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """Lee frame si hay señal; regresa None cuando no trae nada."""

    if cap is None or not cap.isOpened():
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def split_channels(img: np.ndarray) -> ChannelTuple:
    """Separa los canales B, G, R en arreglos independientes."""

    b, g, r = cv2.split(img)
    return b, g, r


def preview_channels(b: np.ndarray, g: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Arma collage B, G, R en falso color para entender el mix."""

    zeros = np.zeros_like(b)
    blue = cv2.merge((b, zeros, zeros))
    green = cv2.merge((zeros, g, zeros))
    red = cv2.merge((zeros, zeros, r))
    return np.hstack((blue, green, red))


CHANNEL_ORDERS: Sequence[Tuple[int, int, int]] = (
    (0, 1, 2),  # BGR
    (0, 2, 1),  # BRG
    (1, 0, 2),  # GBR
    (1, 2, 0),  # GRB
    (2, 0, 1),  # RBG
    (2, 1, 0),  # RGB
)


def reorder_and_gain(
    b: np.ndarray,
    g: np.ndarray,
    r: np.ndarray,
    order_idx: int,
    gains: Tuple[float, float, float],
) -> np.ndarray:
    """Reacomoda canales y aplica ganancia por canal en porcentaje."""

    order = CHANNEL_ORDERS[order_idx % len(CHANNEL_ORDERS)]
    stack = [b, g, r]
    merged = cv2.merge(tuple(stack[i] for i in order))
    gain_arr = np.array(gains, dtype=np.float32).reshape(1, 1, 3) / 100.0
    boosted = cv2.multiply(merged.astype(np.float32), gain_arr)
    return np.clip(boosted, 0, 255).astype(np.uint8)


def posterize(img: np.ndarray, levels: int) -> np.ndarray:
    """Reduce niveles por canal para un look posterizado sabroso."""

    levels = max(1, int(levels))
    if levels == 1:
        return np.zeros_like(img)
    img_f = img.astype(np.float32)
    quantized = np.round(img_f / 255.0 * (levels - 1)) * (255.0 / (levels - 1))
    return np.clip(quantized, 0, 255).astype(np.uint8)


def mosaic(img: np.ndarray, block_size: int) -> np.ndarray:
    """Mosaico express: reducimos resolución por bloques y la subimos de vuelta."""

    block_size = max(1, int(block_size))
    if block_size == 1:
        return img.copy()
    h, w = img.shape[:2]
    trimmed_h = h - (h % block_size)
    trimmed_w = w - (w % block_size)
    core = img[:trimmed_h, :trimmed_w]
    reshaped = core.reshape(
        trimmed_h // block_size,
        block_size,
        trimmed_w // block_size,
        block_size,
        -1,
    )
    block_means = reshaped.mean(axis=(1, 3), keepdims=True).astype(np.uint8)
    mosaic_core = np.broadcast_to(block_means, reshaped.shape).reshape(core.shape)
    result = img.copy()
    result[:trimmed_h, :trimmed_w] = mosaic_core
    return result


def draw_status(img: np.ndarray, text: str) -> np.ndarray:
    """Overlay simple con cv.putText para mostrar estado y fps."""

    display = img.copy()
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        display,
        (pad - 2, pad - 2),
        (pad + text_w + 2, pad + text_h + 6),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        display,
        text,
        (pad, pad + text_h),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return display


def stack_preview(original: np.ndarray, canales: np.ndarray, resultado: np.ndarray) -> np.ndarray:
    """Arma un panel horizontal para echar ojo rápido al pipeline."""

    target_h = 240
    def resize_keep(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        ratio = target_h / img.shape[0]
        new_w = max(1, int(img.shape[1] * ratio))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    o_res = resize_keep(original)
    c_res = resize_keep(canales)
    r_res = resize_keep(resultado)
    return np.hstack((o_res, c_res, r_res))


def toggle_source(idx: int, total: int) -> int:
    """Avanza en el ciclo de fuentes con wrap-around."""

    return (idx + 1) % total


def main() -> None:
    base_image = load_image(DEFAULT_IMAGE_PATH)

    cv2.namedWindow(WINDOW_TRACKBARS)
    cv2.createTrackbar("block_size", WINDOW_TRACKBARS, 8, 50, lambda _x: None)
    cv2.createTrackbar("ch_order", WINDOW_TRACKBARS, 0, 5, lambda _x: None)
    cv2.createTrackbar("gain_B", WINDOW_TRACKBARS, 100, 300, lambda _x: None)
    cv2.createTrackbar("gain_G", WINDOW_TRACKBARS, 100, 300, lambda _x: None)
    cv2.createTrackbar("gain_R", WINDOW_TRACKBARS, 100, 300, lambda _x: None)
    cv2.createTrackbar("posterize_levels", WINDOW_TRACKBARS, 4, 16, lambda _x: None)

    sources = [
        {"label": "IMG", "mode": "image", "source": None},
        {"label": "WEBCAM", "mode": "video", "source": 0},
        {"label": "MÓVIL", "mode": "video", "source": MOBILE_STREAM_URL},
    ]

    capture_state = CaptureState(source=sources[1]["source"])
    current_source_idx = 0
    fps_timer = time.time()
    fps = 0.0

    while True:
        source_info = sources[current_source_idx]
        frame: Optional[np.ndarray]

        if source_info["mode"] == "image":
            frame = base_image.copy()
        else:
            capture_state.source = source_info["source"]
            cap = capture_state.ensure_open()
            frame = read_frame(cap) if cap is not None else None
            if frame is None:
                if isinstance(source_info["source"], str):
                    print("no hay señal del móvil, reintentando…")
                capture_state.cap = open_capture(source_info["source"])
                frame = read_frame(capture_state.cap) if capture_state.cap else None
            if frame is None:
                frame = base_image.copy()

        block_size = max(1, cv2.getTrackbarPos("block_size", WINDOW_TRACKBARS))
        order_idx = cv2.getTrackbarPos("ch_order", WINDOW_TRACKBARS)
        gains = (
            float(cv2.getTrackbarPos("gain_B", WINDOW_TRACKBARS)),
            float(cv2.getTrackbarPos("gain_G", WINDOW_TRACKBARS)),
            float(cv2.getTrackbarPos("gain_R", WINDOW_TRACKBARS)),
        )
        poster_levels = max(1, cv2.getTrackbarPos("posterize_levels", WINDOW_TRACKBARS))

        b, g, r = split_channels(frame)
        if poster_levels > 1:
            b = posterize(b, poster_levels)
            g = posterize(g, poster_levels)
            r = posterize(r, poster_levels)

        merged = reorder_and_gain(b, g, r, order_idx, gains)
        mosaic_img = mosaic(merged, block_size)

        now = time.time()
        dt = now - fps_timer
        fps = 1.0 / dt if dt > 0 else fps
        fps_timer = now

        status_text = f"{source_info['label']} | {fps:4.1f} fps"
        display_result = draw_status(mosaic_img, status_text)

        channel_preview = preview_channels(b, g, r)
        panel = stack_preview(frame, channel_preview, display_result)

        cv2.imshow(WINDOW_ORIGINAL, frame)
        cv2.imshow(WINDOW_CHANNELS, channel_preview)
        cv2.imshow(WINDOW_RESULT, display_result)
        cv2.imshow(WINDOW_TRACKBARS, panel)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("w"), ord("W")):
            current_source_idx = toggle_source(current_source_idx, len(sources))
            if capture_state.cap is not None:
                capture_state.cap.release()
            capture_state.cap = None
            fps_timer = time.time()
        elif key in (ord("s"), ord("S")):
            os.makedirs("out", exist_ok=True)
            out_path = os.path.join("out", "p01_mosaico.png")
            cv2.imwrite(out_path, mosaic_img)
            print(f"resultado guardado en {out_path}")
        elif key in (ord("r"), ord("R")):
            rand_block = random.randint(1, 20)
            rand_order = random.randint(0, len(CHANNEL_ORDERS) - 1)
            rand_gains = [random.randint(60, 220) for _ in range(3)]
            rand_levels = random.randint(1, 8)
            cv2.setTrackbarPos("block_size", WINDOW_TRACKBARS, rand_block)
            cv2.setTrackbarPos("ch_order", WINDOW_TRACKBARS, rand_order)
            cv2.setTrackbarPos("gain_B", WINDOW_TRACKBARS, rand_gains[0])
            cv2.setTrackbarPos("gain_G", WINDOW_TRACKBARS, rand_gains[1])
            cv2.setTrackbarPos("gain_R", WINDOW_TRACKBARS, rand_gains[2])
            cv2.setTrackbarPos("posterize_levels", WINDOW_TRACKBARS, rand_levels)
            print(
                "random preset:",
                {
                    "block": rand_block,
                    "order": rand_order,
                    "gains": rand_gains,
                    "levels": rand_levels,
                },
            )

    if capture_state.cap is not None:
        capture_state.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

