"""Database tab: browse, edit, add and check the pills the app knows."""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .. import pipeline
from ..config import IMAGE_FILETYPES
from ..database import SHAPES, PillDatabase
from ..io_utils import imread
from .common import to_photo

EDIT_FIELDS = ["name", "shape", "colorName", "imprintCode", "dosageMg", "manufacturer", "notes"]
COLUMNS = [("id", 45), ("name", 100), ("shape", 80), ("colorName", 80),
           ("imprintCode", 90), ("dosageMg", 70), ("manufacturer", 120)]


class DatabaseTab(ttk.Frame):
    def __init__(self, parent, state, set_status):
        super().__init__(parent)
        self.state = state
        self.set_status = set_status
        self._photo = None
        self._build()
        self.reload()

    def _build(self):
        bar = ttk.Frame(self, padding=(10, 8))
        bar.pack(fill="x")
        for text, cmd in [("Add pill from current photo", self.add_current), ("Add pill from file...", self.add_file),
                          ("Import folder...", self.import_folder), ("Check database", self.check),
                          ("Delete selected", self.delete)]:
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=(0, 6))
        self.count = ttk.Label(bar, text="")
        self.count.pack(side="right")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=10, pady=4)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew")
        self.tree = ttk.Treeview(left, columns=[c for c, _ in COLUMNS], show="headings", selectmode="browse")
        for col, w in COLUMNS:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        right = ttk.LabelFrame(body, text="Selected pill", padding=8)
        right.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self.thumb = ttk.Label(right, anchor="center")
        self.thumb.grid(row=0, column=0, columnspan=2, pady=(0, 8))
        self.vars = {}
        for i, field in enumerate(EDIT_FIELDS, start=1):
            ttk.Label(right, text=field).grid(row=i, column=0, sticky="e", padx=4, pady=2)
            var = tk.StringVar()
            if field == "shape":
                widget = ttk.Combobox(right, textvariable=var, values=SHAPES, width=22)
            else:
                widget = ttk.Entry(right, textvariable=var, width=25)
            widget.grid(row=i, column=1, sticky="w", pady=2)
            self.vars[field] = var
        self.measured = ttk.Label(right, text="", foreground="#555", justify="left")
        self.measured.grid(row=len(EDIT_FIELDS) + 1, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Button(right, text="Save changes", style="Accent.TButton", command=self.save_changes).grid(
            row=len(EDIT_FIELDS) + 2, column=0, columnspan=2, pady=4)

    # ------------------------------------------------------------ list
    def reload(self):
        self.tree.delete(*self.tree.get_children())
        for row in self.state.db.rows:
            self.tree.insert("", "end", iid=str(row["id"]), values=[row.get(c, "") for c, _ in COLUMNS])
        self.count.configure(text=f"{len(self.state.db.rows)} pills")

    def selected_row(self):
        sel = self.tree.selection()
        return self.state.db.find(sel[0]) if sel else None

    def on_select(self, _event=None):
        row = self.selected_row()
        if not row:
            return
        for field in EDIT_FIELDS:
            self.vars[field].set(row.get(field, ""))
        self.measured.configure(
            text=f"Measured: H={row.get('hue')} S={row.get('saturation')} V={row.get('value')}\n"
                 f"L={row.get('L')} a={row.get('a')} b={row.get('b')}\nAdded: {row.get('dateAdded')}")
        img = imread(self.state.db.image_path(row))
        if img is not None:
            self._photo = to_photo(img, 200)
            self.thumb.configure(image=self._photo)
        else:
            self.thumb.configure(image="", text="(image file missing)")

    # ------------------------------------------------------------ actions
    def save_changes(self):
        row = self.selected_row()
        if not row:
            return
        for field in EDIT_FIELDS:
            row[field] = self.vars[field].get().strip()
        self.state.db.save()
        self.tree.item(str(row["id"]), values=[row.get(c, "") for c, _ in COLUMNS])
        self.set_status(f"Saved {row['name']}.")

    def delete(self):
        row = self.selected_row()
        if row and messagebox.askyesno("Delete", f"Delete {row['name']} from the database?"):
            self.state.db.delete(row["id"])
            self.state.db.features.pop(row["imageFilename"], None)
            self.state.db.save()
            self.state.db.save_cache()
            self.reload()

    def _after_add(self, row):
        self.state.db.save()
        self.state.db.save_cache()
        self.reload()
        self.tree.selection_set(str(row["id"]))
        self.tree.see(str(row["id"]))
        self.set_status(f"Added {row['name']} (shape and colour measured automatically). Fill in imprint, dose and maker.")

    def add_current(self):
        if self.state.path is None:
            messagebox.showinfo("PillScope", "Open an image from a file first (top bar).")
            return
        self._add_path(self.state.path)

    def add_file(self):
        path = filedialog.askopenfilename(filetypes=IMAGE_FILETYPES)
        if path:
            self._add_path(path)

    def _add_path(self, path):
        try:
            self.state.db.ensure_features(self.state.settings)
            row = self.state.db.add_image(path, self.state.settings)
        except Exception as exc:
            messagebox.showerror("PillScope", str(exc))
            return
        self._after_add(row)

    def import_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            self.state.db.ensure_features(self.state.settings)
            n = self.state.db.import_folder(folder, self.state.settings,
                                            progress=lambda i, t: self.set_status(f"Importing {i}/{t}..."))
            self.state.db.save()
            self.state.db.save_cache()
        finally:
            self.config(cursor="")
        self.reload()
        self.set_status(f"Imported {n} images.")

    def check(self):
        problems = self.state.db.validate()
        win = tk.Toplevel(self)
        win.title("Database check")
        win.geometry("640x360")
        text = tk.Text(win, wrap="word", font=("Consolas", 9))
        text.pack(fill="both", expand=True)
        shown = problems[:200]
        text.insert("1.0", "No problems found." if not problems else
                    f"{len(problems)} thing(s) to look at:\n\n" + "\n".join(shown)
                    + ("\n..." if len(problems) > 200 else ""))
        text.configure(state="disabled")
