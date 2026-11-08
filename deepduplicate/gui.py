"""Minimal desktop GUI (Tkinter): pick folders, scan, open the HTML report."""
from __future__ import annotations

import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .embed import BACKBONES
from .pipeline import Config, DuplicateFinder
from .report import write_csv, write_html


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DeepDuplicate"); self.geometry("720x480")
        self.folder, self.ref = tk.StringVar(), tk.StringVar()
        self.backbone = tk.StringVar(value="auto")
        frm = ttk.Frame(self, padding=12); frm.pack(fill="both", expand=True)
        for r, (label, var) in enumerate([("Folder to scan", self.folder), ("Reference (optional)", self.ref)]):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w")
            ttk.Entry(frm, textvariable=var, width=60).grid(row=r, column=1, padx=6, pady=4)
            ttk.Button(frm, text="Browse…", command=lambda v=var: v.set(filedialog.askdirectory() or v.get())).grid(row=r, column=2)
        ttk.Label(frm, text="Backbone").grid(row=2, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.backbone, values=BACKBONES, state="readonly", width=18).grid(row=2, column=1, sticky="w", padx=6)
        self.btn = ttk.Button(frm, text="Find duplicates", command=self.run); self.btn.grid(row=3, column=1, sticky="w", padx=6, pady=8)
        self.log = scrolledtext.ScrolledText(frm, height=18); self.log.grid(row=4, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(4, weight=1); frm.columnconfigure(1, weight=1)

    def say(self, msg: str):
        self.after(0, lambda: (self.log.insert("end", msg + "\n"), self.log.see("end")))

    def run(self):
        if not self.folder.get():
            messagebox.showwarning("DeepDuplicate", "Choose a folder first."); return
        self.btn.state(["disabled"])
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        try:
            self.say(f"Loading backbone '{self.backbone.get()}' …")
            finder = DuplicateFinder(Config(backbone=self.backbone.get()), progress=False)
            self.say(f"Using {finder.embedder.name}. Scanning …")
            ms = finder.find_against(self.folder.get(), self.ref.get()) if self.ref.get() else finder.find_within(self.folder.get())
            out = Path(self.folder.get()) / "deepdup_report"
            write_csv(ms, f"{out}.csv"); write_html(ms, f"{out}.html")
            self.say(f"Found {len(ms)} duplicate pairs → {out}.html")
            webbrowser.open(Path(f"{out}.html").resolve().as_uri())
        except Exception as e:
            self.say(f"Error: {e}")
        finally:
            self.after(0, lambda: self.btn.state(["!disabled"]))


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
