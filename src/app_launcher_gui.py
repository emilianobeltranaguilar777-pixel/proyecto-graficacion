"""
Lanzador GUI con Tkinter (sin dependencias externas).
Ejecuta:  python -m src.app_launcher_tk
"""
from pathlib import Path
import importlib
import pkgutil
import tkinter as tk
from tkinter import ttk, messagebox

from src.core.registry import all_practices

PRACTICAS_DIR = Path(__file__).parent / "practicas"


def load_practices():
    # Importa dinámicamente módulos pXX_*.py para registrar prácticas
    for _, mod, _ in pkgutil.iter_modules([str(PRACTICAS_DIR)]):
        importlib.import_module(f"src.practicas.{mod}")
    return all_practices()


def run_practice(root, fn):
    # Oculta la ventana mientras corre la práctica (OpenCV abre sus propias ventanas)
    root.withdraw()
    try:
        fn()
    except Exception as e:
        messagebox.showerror("Error en la práctica", str(e))
    finally:
        root.deiconify()


def main():
    root = tk.Tk()
    root.title("Proyecto Graficación — Lanzador de Prácticas")
    root.geometry("700x420")

    # Contenedor principal
    frame = ttk.Frame(root, padding=12)
    frame.pack(fill="both", expand=True)

    title = ttk.Label(frame, text="Selecciona una práctica y pulsa Ejecutar",
                      font=("TkDefaultFont", 14))
    title.pack(anchor="w", pady=(0, 8))

    # Lista de prácticas
    columns = ("pid", "title")
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=12)
    tree.heading("pid", text="ID")
    tree.heading("title", text="Título")
    tree.column("pid", width=80, anchor="w")
    tree.column("title", width=520, anchor="w")

    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)

    tree.pack(side="left", fill="both", expand=True)
    vsb.pack(side="left", fill="y", padx=(4, 0))

    # Barra de botones
    btns = ttk.Frame(frame)
    btns.pack(side="right", fill="y", padx=(8, 0))

    def refresh():
        for row in tree.get_children():
            tree.delete(row)
        practices[:] = load_practices()
        if not practices:
            messagebox.showinfo(
                "Sin prácticas",
                "No hay prácticas registradas.\n"
                "Agrega archivos pXX_*.py en src/practicas/ y pulsa Refrescar."
            )
        else:
            for pid, title, _ in practices:
                tree.insert("", "end", values=(pid, title))

    def run_selected():
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("Selecciona una práctica", "Selecciona una fila de la lista.")
            return
        pid = tree.item(sel[0], "values")[0]
        for ppid, _, fn in practices:
            if ppid == pid:
                run_practice(root, fn)
                return

    ttk.Button(btns, text="Ejecutar", command=run_selected).pack(fill="x", pady=(0, 6))
    ttk.Button(btns, text="Refrescar", command=refresh).pack(fill="x", pady=(0, 6))
    ttk.Button(btns, text="Salir", command=root.destroy).pack(fill="x")

    # Carga inicial
    practices = []
    refresh()

    root.mainloop()


if __name__ == "__main__":
    main()
