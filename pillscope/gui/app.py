"""Main window."""
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..config import IMAGE_FILETYPES, IMAGE_PATTERNS, SAMPLES_DIR
from .common import AppState
from .database_tab import DatabaseTab
from .identify_tab import IdentifyTab
from .panels import ALL_PANELS


class PillScopeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PillScope - pill identification with classical computer vision")
        self.geometry("1280x860")
        self.minsize(1100, 760)
        self.state = AppState()
        self._style()
        self._build()
        self.after(200, self.prepare_database)

    def _style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), foreground="white", background="#2563eb")
        style.map("Accent.TButton", background=[("active", "#1d4ed8")])
        style.configure("Header.TLabel", font=("Segoe UI", 15, "bold"))

    def _build(self):
        header = ttk.Frame(self, padding=(12, 10))
        header.pack(fill="x")
        ttk.Label(header, text="PillScope", style="Header.TLabel").pack(side="left")
        ttk.Button(header, text="Open image...", command=self.open_image).pack(side="left", padx=(20, 6))
        samples = sorted(p.name for pat in IMAGE_PATTERNS for p in SAMPLES_DIR.glob(pat)) if SAMPLES_DIR.exists() else []
        self.sample_var = tk.StringVar()
        box = ttk.Combobox(header, textvariable=self.sample_var, values=samples, width=42, state="readonly")
        box.pack(side="left")
        box.set("or pick a sample photo..." if samples else "(no sample photos found)")
        box.bind("<<ComboboxSelected>>", self.open_sample)
        self.path_label = ttk.Label(header, text="No image loaded", foreground="#555")
        self.path_label.pack(side="left", padx=12)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=(8, 2)).pack(side="bottom", fill="x")

        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=6, pady=(0, 4))
        self.identify = IdentifyTab(self.tabs, self.state, self.set_status)
        self.tabs.add(self.identify, text="  Identify  ")

        explore = ttk.Notebook(self.tabs)
        for panel_cls in ALL_PANELS:
            explore.add(panel_cls(explore, self.state), text=panel_cls.title)
        self.tabs.add(explore, text="  Explore the labs  ")

        self.database = DatabaseTab(self.tabs, self.state, self.set_status)
        self.tabs.add(self.database, text="  Database  ")
        self.state.subscribe(self._update_path_label)

    # ------------------------------------------------------------ actions
    def set_status(self, text):
        self.status.set(text)
        self.update_idletasks()

    def _update_path_label(self):
        p = self.state.path
        self.path_label.configure(text=Path(p).name if p else "(image from memory)")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=IMAGE_FILETYPES, initialdir=str(SAMPLES_DIR))
        if path:
            self._load(path)

    def open_sample(self, _event=None):
        self._load(SAMPLES_DIR / self.sample_var.get())

    def _load(self, path):
        try:
            self.state.load_path(path)
        except Exception as exc:
            messagebox.showerror("PillScope", str(exc))

    def _progress(self, done, total):
        if done % 20 == 0 or done == total:
            self.set_status(f"First start: preparing database pills ({done}/{total})...")

    def prepare_database(self):
        db = self.state.db
        if not db.rows:
            self.set_status("The database is empty. Run scripts/build_database.py or add pills in the Database tab.")
            return
        self.config(cursor="watch")
        try:
            db.ensure_features(self.state.settings, progress=self._progress)
        finally:
            self.config(cursor="")
        self.set_status(f"Database ready: {len(db.rows)} pills. Open a photo to identify it.")


def main():
    PillScopeApp().mainloop()


if __name__ == "__main__":
    main()
