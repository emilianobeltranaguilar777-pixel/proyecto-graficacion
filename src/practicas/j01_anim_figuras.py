#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g01 — Animación de figuras geométricas (juego visual sin cámara)
Controles: [ESPACIO] escena · [S] captura · [Q]/ESC salir
"""
import os, time, math, cv2 as cv, numpy as np
from pathlib import Path

try:
    from src.core.registry import register
except Exception:
    from core.registry import register

@register("g01", "Animación de figuras geométricas (juego)")
def run(width=960, height=720):
    # evitar error Qt/Wayland
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    Path("assets/capturas").mkdir(parents=True, exist_ok=True)

    win = "g01_anim_figuras"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.resizeWindow(win, width, height)

    scene = 0
    t0 = time.perf_counter()

    def fondo(t):
        yv = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        c1, c2 = np.array([40, 30, 60]), np.array([70, 40, 100])
        c2 = c2 + 20*np.sin(t*0.3)
        bg = (c1*(1-yv) + c2*yv).astype(np.uint8)
        return np.repeat(bg, width, axis=1).reshape(height, width, 3)

    def escena1(img, t):
        cx, cy = width//2, height//2
        for i in range(12):
            r = int(80 + 50*math.sin(t + i))
            ang = t + i*math.pi/6
            x = int(cx + 200*math.cos(ang))
            y = int(cy + 200*math.sin(ang))
            cv.circle(img, (x, y), r//6, (80+10*i, 200-10*i, 255), -1, cv.LINE_AA)

    def escena2(img, t):
        for i in range(8):
            cx, cy = width//2, height//2
            a = t*40 + i*45
            size = 60 + 10*i
            pts = np.array([
                [cx+size*math.cos(math.radians(a+0)),  cy+size*math.sin(math.radians(a+0))],
                [cx+size*math.cos(math.radians(a+120)),cy+size*math.sin(math.radians(a+120))],
                [cx+size*math.cos(math.radians(a+240)),cy+size*math.sin(math.radians(a+240))]
            ], np.int32)
            cv.polylines(img,[pts],True,(255-20*i,80+20*i,200),2,cv.LINE_AA)

    def escena3(img, t):
        for i in range(5):
            n = 5+i
            r = 80 + 30*i
            pts = []
            for k in range(n):
                th = t*0.5 + 2*math.pi*k/n
                pts.append([int(width/2+r*math.cos(th)), int(height/2+r*math.sin(th))])
            cv.polylines(img,[np.array(pts,np.int32)],True,(100+30*i,200-20*i,255),2,cv.LINE_AA)

    escenas = [escena1, escena2, escena3]

    print("🎮 g01 — Animación de figuras geométricas (sin cámara)")
    print("Controles: [ESPACIO] siguiente escena | [S] captura | [Q]/ESC salir")

    while True:
        t = time.perf_counter() - t0
        frame = fondo(t)
        escenas[scene](frame, t)
        cv.putText(frame, f"Escena {scene+1}", (20,40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
        cv.imshow(win, frame)

        k = cv.waitKey(1) & 0xFF
        if k in (27, ord('q')): break
        elif k == ord(' '): scene = (scene+1) % len(escenas)
        elif k in (ord('s'), ord('S')):
            fname = f"assets/capturas/g01_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv.imwrite(fname, frame)
            cv.displayOverlay(win, f"Captura guardada: {fname}", 1500)

    cv.destroyAllWindows()

if __name__ == "__main__":
    run()
