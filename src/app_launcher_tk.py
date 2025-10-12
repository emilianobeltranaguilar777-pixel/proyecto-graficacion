import os
import pkgutil
import importlib
import tkinter as tk
from tkinter import ttk, messagebox

# Siempre usar el paquete "src"
from src.core.registry import all_practices

# Autoimport de módulos pXX_*.py dentro de src/practicas
def _autoimport_practicas():
    import src.practicas as practicas_pkg
    for mod in pkgutil.iter_modules(practicas_pkg.__path__):
        name = mod.name
        # Acepta p00, p01..., p99_*.py
        if len(name) >= 3 and name[0] == 'p' and name[1:3].isdigit():
            importlib.import_module(f"src.practicas.{name}")

def main():
    _autoimport_practicas()  # llena el registro

    root = tk.Tk()
    root.title("Proyecto Graficación – Lanzador de Prácticas")

    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    cols = ("ID", "Título")
    tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=280 if c == "Título" else 80, anchor="w")
    tree.pack(fill="both", expand=True)

    # Obtener prácticas y ordenarlas por ID (pXX)
    items = sorted(all_practices(), key=lambda x: x[0])  # (id, title, func)

    for pid, title, _func in items:
        tree.insert("", "end", iid=pid, values=(pid, title))

    btns = ttk.Frame(frame)
    btns.pack(fill="x", pady=(8,0))

    def run_selected():
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("Aviso", "Selecciona una práctica.")
            return
        pid = sel[0]
        # Buscar función por ID
        for _pid, _title, fn in items:
            if _pid == pid:
                try:
                    fn()
                except Exception as e:
                    messagebox.showerror("Error ejecutando práctica", str(e))
                return

    run_btn = ttk.Button(btns, text="Ejecutar", command=run_selected)
    run_btn.pack(side="right")

    root.mainloop()

if __name__ == "__main__":
    # En Wayland necesitas esto exportado antes de lanzar Python:
    #   export QT_QPA_PLATFORM=xcb
    main()
