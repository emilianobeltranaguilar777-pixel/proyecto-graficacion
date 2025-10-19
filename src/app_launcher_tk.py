#!/usr/bin/env python3
"""
Launcher GUI (Neon – diseño 2 columnas) para proyecto-graficacion
- Fondo auto-fit a tamaño de teléfono (configurable por env PG_MAX_W/PG_MAX_H)
- FILTROS/JUEGOS lado a lado, SELECCIONADO ancho y dos botones abajo
- Listbox con margen interior; texto SELECCIONADO sobre el canvas (con ancho fijo)
- Compatibilidad con registry viejo (all_practices) y nuevo (REGISTRY/get_registry)

Uso:
  export QT_QPA_PLATFORM=xcb
  # (opcional) rutas de imágenes:
  #   PG_UI_BG, PG_UI_PAC_HAPPY, PG_UI_PAC_ANGRY
  # (opcional) tamaño de ventana:
  #   PG_MAX_W=360 PG_MAX_H=720
  python -m src.app_launcher_tk
"""
from __future__ import annotations
import os
import sys
import threading
import importlib
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from tkinter import font as tkfont
from typing import Tuple

try:
    from PIL import Image, ImageTk, ImageOps  # type: ignore
except Exception:
    print("[WARN] Falta Pillow. Instala con: pip install pillow", file=sys.stderr)
    raise

# --- Paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "assets" / "ui"
BG_PATH = os.environ.get("PG_UI_BG", str(ASSETS / "layout_neon.png"))
PAC_HAPPY = os.environ.get("PG_UI_PAC_HAPPY", str(ASSETS / "pacman_happy.png"))
PAC_ANGRY = os.environ.get("PG_UI_PAC_ANGRY", str(ASSETS / "pacman_angry.png"))

# --- Registry bridge -------------------------------------------------------
REGISTRY: dict[str, dict] = {}

def _safe_title_from_module(mod, default_title: str):
    # 1) __title__ si existe, 2) primera línea del docstring, 3) default
    title = getattr(mod, "__title__", None)
    if title:
        return str(title).strip()
    doc = getattr(mod, "__doc__", "") or ""
    line = doc.strip().splitlines()[0].strip() if doc.strip() else ""
    return line if line else default_title

def _fallback_scan_modules_into_registry():
    """
    Si el registry no trajo todas las prácticas, escanea src/practicas/{pXX_*.py,jXX_*.py},
    importa los módulos y, si tienen run() callable, los agrega a REGISTRY con tag.
    """
    global REGISTRY
    pract_dir = REPO_ROOT / "src" / "practicas"
    sys.path.insert(0, str(REPO_ROOT))
    pkg_base = "src.practicas"

    for pattern in ("p[0-9][0-9]_*.py", "j[0-9][0-9]_*.py"):
        for py in sorted(pract_dir.glob(pattern)):
            pid = py.stem.split("_")[0]          # p02 / j01
            if pid in REGISTRY:
                continue
            modname = f"{pkg_base}.{py.stem}"
            try:
                mod = importlib.import_module(modname)
            except Exception as e:
                print(f"[WARN] Fallback no pudo importar {modname}: {e}")
                continue
            fn = getattr(mod, "run", None)
            if callable(fn):
                title = _safe_title_from_module(mod, f"{pid} — práctica")
                tag = "juegos" if pid.lower().startswith("j") else "filtros"
                REGISTRY[pid] = {"title": title, "callable": fn, "tag": tag}

def _registry_autoimport_fallback():
    """Importa src/practicas/{pXX_*.py,jXX_*.py} para forzar el registro."""
    pract_dir = REPO_ROOT / "src" / "practicas"
    sys.path.insert(0, str(REPO_ROOT))
    pkg_base = "src.practicas"
    for pattern in ("p[0-9][0-9]_*.py", "j[0-9][0-9]_*.py"):
        for py in sorted(pract_dir.glob(pattern)):
            modname = f"{pkg_base}.{py.stem}"
            try:
                importlib.import_module(modname)
            except Exception as e:
                print(f"[WARN] No se pudo importar {modname}: {e}")

def load_registry():
    """Carga el registro desde distintas APIs soportadas."""
    global REGISTRY
    sys.path.insert(0, str(REPO_ROOT))
    try:
        reg = importlib.import_module("src.core.registry")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo importar registry: {e}")
        raise

    # Intentar discover() si existe; si no, autoimport fallback
    if hasattr(reg, "discover"):
        try:
            reg.discover()
        except Exception as e:
            print(f"[WARN] registry.discover() falló: {e}")
    else:
        _registry_autoimport_fallback()

    # 1) API nueva: dict REGISTRY
    if hasattr(reg, "REGISTRY"):
        REGISTRY = getattr(reg, "REGISTRY")
        return

    # 2) API nueva: get_registry()
    if hasattr(reg, "get_registry"):
        REGISTRY = dict(getattr(reg, "get_registry")())
        return

    # 3) API vieja: all_practices() -> [(pid, title, fn), ...]
    if hasattr(reg, "all_practices"):
        try:
            items = list(reg.all_practices())
        except Exception as e:
            raise RuntimeError(f"all_practices() falló: {e}")
        REGISTRY = {
            pid: {
                "title": title,
                "callable": fn,
                "tag": ("juegos" if pid.lower().startswith("j") else "filtros"),
            }
            for (pid, title, fn) in items
        }
        return

    raise RuntimeError("registry no expone REGISTRY, get_registry() ni all_practices()")

# --- Geometry helpers ------------------------------------------------------
def rect_pct(w: int, h: int, x: float, y: float, pw: float, ph: float) -> Tuple[int, int, int, int]:
    """Devuelve (x1,y1,x2,y2) en píxeles a partir de porcentajes del fondo."""
    x1 = int(w * x)
    y1 = int(h * y)
    x2 = int(w * (x + pw))
    y2 = int(h * (y + ph))
    return x1, y1, x2, y2

def inset(rect: Tuple[int, int, int, int], frac: float = 0.12) -> Tuple[int, int, int, int]:
    """Encoge un rectángulo por un porcentaje (margen interior)."""
    x1, y1, x2, y2 = rect
    dx = int((x2 - x1) * frac)
    dy = int((y2 - y1) * frac)
    return x1 + dx, y1 + dy, x2 - dx, y2 - dy

class NeonLauncher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BELRAN — Launcher")
        self.configure(bg="#000000")

        # --- Canvas + fondo (auto-fit a teléfono) --------------------------
        MAX_W = int(os.environ.get("PG_MAX_W", 360))   # ancho máximo ventana
        MAX_H = int(os.environ.get("PG_MAX_H", 720))   # alto máximo ventana
        bg_img = Image.open(BG_PATH)
        bg_img = ImageOps.contain(bg_img, (MAX_W, MAX_H), Image.LANCZOS)

        self.W, self.H = bg_img.size
        self.geometry(f"{self.W}x{self.H}")
        self.resizable(False, False)

        self.canvas = tk.Canvas(self, width=self.W, height=self.H, highlightthickness=0, bd=0, bg="#000")
        self.canvas.pack(fill="both", expand=True)
        self._bg_tk = ImageTk.PhotoImage(bg_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self._bg_tk)

        # --- Layout (porcentajes afinados al mock) -------------------------
        self.RECT_FILTROS = rect_pct(self.W, self.H, 0.099, 0.30, 0.44, 0.23)
        self.RECT_JUEGOS  = rect_pct(self.W, self.H, 0.450, 0.30, 0.45, 0.23)
        self.RECT_SEL     = rect_pct(self.W, self.H, 0.055, 0.57, 0.90, 0.14)
        self.RECT_BTN_L   = rect_pct(self.W, self.H, 0.130,  0.73, 0.40, 0.17)
        self.RECT_BTN_R   = rect_pct(self.W, self.H, 0.475, 0.73, 0.40, 0.17)

        self.RECT_FILTROS_IN = inset(self.RECT_FILTROS, 0.20)
        self.RECT_JUEGOS_IN  = inset(self.RECT_JUEGOS,  0.20)
        self.RECT_SEL_IN     = inset(self.RECT_SEL,     0.12)
        self.RECT_BTN_L_IN   = inset(self.RECT_BTN_L,   0.18)
        self.RECT_BTN_R_IN   = inset(self.RECT_BTN_R,   0.18)

        # Listas
        self.lb_filtros = self._make_listbox(self.RECT_FILTROS_IN)
        self.lb_juegos  = self._make_listbox(self.RECT_JUEGOS_IN)
        self.lb_filtros.bind("<<ListboxSelect>>", lambda e: self._on_select("filtros"))
        self.lb_juegos.bind("<<ListboxSelect>>", lambda e: self._on_select("juegos"))

        # Texto SELECCIONADO
        self.sel_var = tk.StringVar(value="— nada seleccionado —")
        x1, y1, x2, y2 = self.RECT_SEL_IN
        self._sel_w = (x2 - x1) - 6
        self._sel_h = (y2 - y1) - 4
        self.sel_font = tkfont.Font(family="Arial", size=12, weight="bold")

        cx, cy = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.sel_text_id = self.canvas.create_text(
            (cx, cy),
            text="",
            fill="#FFFFFF",
            font=self.sel_font,
            width=self._sel_w,
            anchor="center",
            justify="center",
        )
        self._set_selection_text(self.sel_var.get())

        # Botones
        def scale_to_rect_keep_ratio(img_path, rect, fill_pct=0.60):
            img = Image.open(img_path).convert("RGBA")
            x1, y1, x2, y2 = rect
            rw, rh = (x2 - x1), (y2 - y1)
            target = (int(rw * fill_pct), int(rh * fill_pct))
            img = ImageOps.contain(img, target, Image.LANCZOS)
            return ImageTk.PhotoImage(img)

        self._img_happy = scale_to_rect_keep_ratio(PAC_HAPPY, self.RECT_BTN_L_IN, fill_pct=1)
        self._img_angry = scale_to_rect_keep_ratio(PAC_ANGRY, self.RECT_BTN_R_IN, fill_pct=1)

        self.btn_exec = tk.Button(
            self, image=self._img_happy, command=self.execute_selected,
            bd=0, highlightthickness=0, cursor="hand2", relief="flat", bg="#000"
        )
        self.btn_clear = tk.Button(
            self, image=self._img_angry, command=self.clear_selection,
            bd=0, highlightthickness=0, cursor="hand2", relief="flat", bg="#000"
        )

        def center(rect):
            x1, y1, x2, y2 = rect
            return ((x1 + x2) // 2, (y1 + y2) // 2)

        cx, cy = center(self.RECT_BTN_L_IN)
        self.canvas.create_window((cx, cy), anchor="center", window=self.btn_exec)
        cx, cy = center(self.RECT_BTN_R_IN)
        self.canvas.create_window((cx, cy), anchor="center", window=self.btn_clear)

        # Poblar prácticas
        self.populate_lists()
        self.selected_key: str | None = None

    # ---- UI builders ----
    def _make_listbox(self, rect):
        x1, y1, x2, y2 = rect
        frame = tk.Frame(self, bg="#000000")
        lb = tk.Listbox(
            frame,
            activestyle="dotbox",
            fg="#8adcff",
            bg="#0d0f14",
            selectbackground="#1a8fe6",
            selectforeground="#000000",
            highlightthickness=0,
            bd=0,
            font=("Arial", 11, "bold")
        )
        lb.pack(fill="both", expand=True)
        self.canvas.create_window(
            (x1, y1), anchor="nw", width=(x2 - x1), height=(y2 - y1), window=frame
        )
        return lb

    # ---- Registry & population ----
    def populate_lists(self):
        load_registry()
        _fallback_scan_modules_into_registry()
        filtros, juegos = [], []
        for key, item in sorted(REGISTRY.items()):
            title = item.get("title", key)
            tag = item.get("tag", "filtros")
            entry = f"{key} — {title}"
            (juegos if tag == "juegos" else filtros).append(entry)
        self.lb_filtros.delete(0, tk.END)
        self.lb_juegos.delete(0, tk.END)
        for e in filtros:
            self.lb_filtros.insert(tk.END, e)
        for e in juegos:
            self.lb_juegos.insert(tk.END, e)

    # ---- Selection & actions ----
    def _on_select(self, which):
        lb = self.lb_filtros if which == "filtros" else self.lb_juegos
        (self.lb_juegos if which == "filtros" else self.lb_filtros).selection_clear(0, tk.END)
        idxs = lb.curselection()
        if not idxs:
            return
        text = lb.get(idxs[0])
        key = text.split(" — ")[0]
        self.selected_key = key
        self.sel_var.set(text)
        self._set_selection_text(self.sel_var.get())

    def clear_selection(self):
        self.lb_filtros.selection_clear(0, tk.END)
        self.lb_juegos.selection_clear(0, tk.END)
        self.selected_key = None
        self.sel_var.set("— nada seleccionado —")
        self._set_selection_text(self.sel_var.get())

    def execute_selected(self):
        if not self.selected_key:
            messagebox.showinfo("Ejecutar", "Selecciona una práctica primero.")
            return
        item = REGISTRY.get(self.selected_key, {})
        func = item.get("callable") or item.get("fn")
        if not callable(func):
            messagebox.showerror("Error", f"La práctica {self.selected_key} no tiene callable válido.")
            return
        # Forzar XCB siempre, evita crash del plugin Wayland de OpenCV
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        threading.Thread(target=lambda: self._run_safe(func), daemon=True).start()

    # ---- Helper: ejecuta práctica en hilo aislado ----
    def _run_safe(self, func):
        try:
            func()
        except Exception as e:
            messagebox.showerror("Error en práctica", f"Ocurrió un error:\n{e}")

    # ---- Helper: wrap + recorte con “…” dentro del rectángulo ----
    def _set_selection_text(self, raw_text: str):
        max_w = self._sel_w
        max_h = self._sel_h
        fnt = self.sel_font

        words = raw_text.split()
        lines = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if fnt.measure(test) <= max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)

        line_h = fnt.metrics("linespace") or 14
        max_lines = max(1, max_h // line_h)

        if len(lines) > max_lines:
            keep = lines[:max_lines]
            last = keep[-1]
            ell = "…"
            while last and fnt.measure(last + ell) > max_w:
                last = last[:-1]
            keep[-1] = (last + ell) if last else ell
            lines = keep

        display = "\n".join(lines) if lines else raw_text
        self.canvas.itemconfigure(self.sel_text_id, text=display)

# --- Main ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        app = NeonLauncher()
        app.mainloop()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
