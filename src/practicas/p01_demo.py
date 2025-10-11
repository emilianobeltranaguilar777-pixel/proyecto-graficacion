import cv2
import numpy as np
from src.core.registry import register

@register("p01", "Demo: Hola OpenCV")
def run():
    win = "Demo p01"
    img = np.full((300, 600, 3), 30, np.uint8)
    cv2.putText(img, "Hola OpenCV [Q para salir]", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow(win, img)
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q'), ord('Q')):
            break
    cv2.destroyWindow(win)
