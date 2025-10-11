"""
Lanzador de prácticas con menú (OpenCV).
Teclas:
  ↑/↓ navegan | ENTER ejecuta | Q/ESC salir
"""
import cv2
import numpy as np
from core.registry import all_practices

WIN = "Menu | Proyecto Graficación"

def render_menu(items, idx):
    h, w = 520, 840
    img = np.full((h, w, 3), 24, np.uint8)
    cv2.putText(img, "Selecciona una práctica y presiona ENTER",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    y = 90
    for i, (pid, title, _) in enumerate(items):
        color = (255, 255, 255) if i == idx else (170, 170, 170)
        prefix = "▶ " if i == idx else "   "
        cv2.putText(img, f"{prefix}{pid}: {title}", (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 40
    cv2.putText(img, "↑/↓ mover | ENTER ejecutar | Q/ESC salir",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1)
    return img

def main():
    items = all_practices()
    if not items:
        raise SystemExit("No hay prácticas registradas todavía.")
    idx = 0
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 560)

    while True:
        cv2.imshow(WIN, render_menu(items, idx))
        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord('q'), ord('Q')):
            break
        elif k in (ord('\r'), 10, 13):  # ENTER
            cv2.destroyWindow(WIN)
            try:
                items[idx][2]()  # ejecuta práctica
            except Exception as e:
                print("Error en la práctica:", e)
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, 960, 560)
        elif k in (82, ord('k')):  # ↑ (82 en waitKey), o 'k' estilo vim
            idx = (idx - 1) % len(items)
        elif k in (84, ord('j')):  # ↓ (84 en waitKey), o 'j'
            idx = (idx + 1) % len(items)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
