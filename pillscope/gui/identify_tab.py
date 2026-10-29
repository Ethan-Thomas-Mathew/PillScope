"""The main tab: photo in -> pipeline stages -> answer."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from .. import pipeline, report
from ..config import OUTPUT_DIR
from ..io_utils import imread
from .common import to_photo

STAGE_SIZE = 210
VERDICT_COLORS = {"match": "#1b7f3b", "possible": "#b36b00", "unknown": "#b00020"}


class IdentifyTab(ttk.Frame):
    def __init__(self, parent, state, set_status):
        super().__init__(parent)
        self.state = state
        self.set_status = set_status
        self.analysis = None
        self.matches = []
        self._ref_cache = {}
        self._photos = []                      # keep references so Tk does not drop the images
        self._build()
        state.subscribe(self.on_image_changed)

    # ------------------------------------------------------------ layout
    def _build(self):
        s = self.state.settings
        self.var_denoise = tk.BooleanVar(value=s.denoise)
        self.var_clahe = tk.BooleanVar(value=s.clahe)
        self.var_seg = tk.StringVar(value=s.seg_method)
        self.var_ratio = tk.DoubleVar(value=s.sift_ratio)

        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(fill="x")
        ttk.Button(bar, text="Identify pill", style="Accent.TButton", command=self.run).pack(side="left")
        ttk.Button(bar, text="Save report picture...", command=self.save_report).pack(side="left", padx=8)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Checkbutton(bar, text="Median denoise", variable=self.var_denoise).pack(side="left", padx=4)
        ttk.Checkbutton(bar, text="CLAHE contrast", variable=self.var_clahe).pack(side="left", padx=4)
        ttk.Label(bar, text="Pill outline:").pack(side="left", padx=(12, 2))
        ttk.Combobox(bar, textvariable=self.var_seg, values=["threshold", "region_growing"],
                     width=14, state="readonly").pack(side="left")
        ttk.Label(bar, text="SIFT ratio:").pack(side="left", padx=(12, 2))
        ttk.Spinbox(bar, from_=0.5, to=0.95, increment=0.05, textvariable=self.var_ratio, width=5).pack(side="left")

        # pipeline strip
        strip = ttk.LabelFrame(self, text="What the pipeline sees", padding=6)
        strip.pack(fill="x", padx=10, pady=4)
        self.stage_labels, self.stage_captions = [], []
        titles = ["1. Photo + pill outline", "2. Preprocessed", "3. Canny edges", "4. SIFT keypoints"]
        for i, title in enumerate(titles):
            cell = ttk.Frame(strip)
            cell.grid(row=0, column=i, padx=6)
            lbl = ttk.Label(cell, anchor="center")
            lbl.pack()
            ttk.Label(cell, text=title, font=("Segoe UI", 9, "bold")).pack()
            self.stage_labels.append(lbl)
        strip.columnconfigure(tuple(range(4)), weight=1)

        # lower area
        lower = ttk.Frame(self)
        lower.pack(fill="both", expand=True, padx=10, pady=4)
        lower.columnconfigure(1, weight=1)
        lower.rowconfigure(0, weight=1)

        details = ttk.LabelFrame(lower, text="Pill properties", padding=8)
        details.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        self.details = tk.Text(details, width=34, height=14, relief="flat", font=("Consolas", 9),
                               background="#f6f6f6", state="disabled")
        self.details.pack(fill="both", expand=True)

        result = ttk.LabelFrame(lower, text="Answer", padding=8)
        result.grid(row=0, column=1, sticky="nsew")
        result.columnconfigure(0, weight=1)
        result.rowconfigure(2, weight=1)
        self.verdict = ttk.Label(result, text="Open an image to begin.", font=("Segoe UI", 12, "bold"), wraplength=700)
        self.verdict.grid(row=0, column=0, sticky="w")
        cols = ("rank", "name", "score", "sift", "shape", "color")
        self.tree = ttk.Treeview(result, columns=cols, show="headings", height=5)
        for col, text, w in [("rank", "#", 30), ("name", "Pill", 100), ("score", "Score", 60),
                             ("sift", "SIFT matches", 90), ("shape", "Shape", 80), ("color", "Colour", 80)]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor="center")
        self.tree.grid(row=1, column=0, sticky="ew", pady=6)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.match_label = ttk.Label(result, anchor="center")
        self.match_label.grid(row=2, column=0, sticky="nsew")
        self.match_caption = ttk.Label(result, text="", foreground="#555")
        self.match_caption.grid(row=3, column=0, sticky="w")

    # ------------------------------------------------------------ actions
    def apply_controls(self):
        s = self.state.settings
        s.denoise, s.clahe = self.var_denoise.get(), self.var_clahe.get()
        s.seg_method, s.sift_ratio = self.var_seg.get(), round(float(self.var_ratio.get()), 2)

    def on_image_changed(self):
        self._show_stage(0, self.state.image)
        for i in (1, 2, 3):
            self.stage_labels[i].configure(image="")
        self._clear_results()
        if self.state.db.rows:
            self.run()

    def _clear_results(self):
        self.analysis, self.matches = None, []
        self.tree.delete(*self.tree.get_children())
        self.match_label.configure(image="")
        self.match_caption.configure(text="")
        self._set_details([])

    def run(self):
        img = self.state.image
        if img is None:
            messagebox.showinfo("PillScope", "Open an image first (top bar).")
            return
        if not self.state.db.rows:
            messagebox.showinfo("PillScope", "The database is empty. Add pills in the Database tab first.")
            return
        self.apply_controls()
        s = self.state.settings
        self.config(cursor="watch")
        self.set_status("Analysing photo...")
        self.update_idletasks()
        try:
            n = self.state.db.ensure_features(s)
            if n:
                self._ref_cache.clear()
            a = pipeline.analyze(img, s)
            matches = pipeline.identify(a, self.state.db, s)
        except Exception as exc:                                     # show, don't crash the window
            messagebox.showerror("PillScope", f"Something went wrong:\n{exc}")
            return
        finally:
            self.config(cursor="")
        self.analysis, self.matches = a, matches
        self._show_results(a, matches)
        self.set_status(pipeline.verdict(matches)[1])

    def _show_stage(self, index, img, rgb=False):
        photo = to_photo(img, STAGE_SIZE, rgb=rgb)
        self._photos.append(photo)
        self._photos = self._photos[-60:]
        self.stage_labels[index].configure(image=photo)

    def _show_results(self, a, matches):
        self._show_stage(0, report.mask_overlay(a), rgb=True)
        self._show_stage(1, a.gray)
        self._show_stage(2, a.edge_map)
        self._show_stage(3, report.keypoint_view(a), rgb=True)
        self._set_details(report.describe(a))

        kind, sentence = pipeline.verdict(matches)
        self.verdict.configure(text=sentence, foreground=VERDICT_COLORS[kind])
        self.tree.delete(*self.tree.get_children())
        for i, m in enumerate(matches, 1):
            self.tree.insert("", "end", iid=str(i - 1), values=(
                i, m.row["name"], f"{m.score * 100:.1f}", m.inliers, m.row["shape"], m.row["colorName"]))
        if matches:
            self.tree.selection_set("0")

    def _set_details(self, lines):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", "\n".join(lines))
        self.details.configure(state="disabled")

    def on_select(self, _event=None):
        sel = self.tree.selection()
        if not sel or self.analysis is None:
            return
        match = self.matches[int(sel[0])]
        name = match.row["imageFilename"]
        if name not in self._ref_cache:
            ref_img = imread(self.state.db.image_path(match.row))
            if ref_img is None:
                self.match_caption.configure(text="Reference image file is missing.")
                return
            self._ref_cache[name] = pipeline.analyze(ref_img, self.state.settings)
        view = pipeline.draw_match_view(self.analysis, self._ref_cache[name], self.state.settings.sift_ratio)
        photo = to_photo(view, 720, 300, rgb=True)
        self._photos.append(photo)
        self.match_label.configure(image=photo)
        self.match_caption.configure(
            text=f"Your photo (left) vs {match.row['name']} (right). Green lines join matching SIFT keypoints.")

    def save_report(self):
        if self.analysis is None:
            messagebox.showinfo("PillScope", "Identify a pill first.")
            return
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(defaultextension=".png", initialdir=str(OUTPUT_DIR),
                                            initialfile="report.png", filetypes=[("PNG image", "*.png")])
        if not path:
            return
        fig = report.build_report_figure(self.analysis, self.matches, self.state.settings, self.state.db)
        fig.savefig(path, bbox_inches="tight")
        self.set_status(f"Saved report: {path}")
