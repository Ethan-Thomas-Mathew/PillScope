"""Explore tab: one small interactive panel for each lab, all working on the current photo."""
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .. import basic_ops, color, edges, features, frequency, sampling, segmentation, spatial


def show(ax, img, title, cmap=None, bgr=True):
    """Draw an image on an axis. Grey images use a grey colour map."""
    if img.ndim == 2:
        ax.imshow(img, cmap=cmap or "gray", vmin=0, vmax=255)
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if bgr else img)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def gray_of(img):
    return basic_ops.to_gray(img)


def shrink(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    f = max_side / float(max(h, w))
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


class Panel(ttk.Frame):
    """Base class: a row of controls on top, a matplotlib figure below."""
    title = "Panel"
    help_text = ""

    def __init__(self, parent, state):
        super().__init__(parent)
        self.state = state
        self._after = None
        self._dirty = True
        self.controls = ttk.Frame(self, padding=(8, 6))
        self.controls.pack(side="top", fill="x")
        self.note = ttk.Label(self, text=self.help_text, wraplength=1100, foreground="#555", padding=(8, 2))
        self.note.pack(side="bottom", fill="x")
        self.fig = Figure(figsize=(9, 5.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.build_controls(self.controls)
        state.subscribe(self.on_image_changed)
        self.bind("<Map>", lambda e: self._refresh_if_dirty())

    # --- to override
    def build_controls(self, frame):
        pass

    def draw(self, fig, img):
        raise NotImplementedError

    # --- plumbing
    def on_image_changed(self):
        self._dirty = True
        if self.winfo_ismapped():
            self.refresh()

    def _refresh_if_dirty(self):
        if self._dirty:
            self.refresh()

    def changed(self, *_):
        """Called by controls; waits a moment so dragging a slider is not laggy."""
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(150, self.refresh)

    def refresh(self):
        self._after, self._dirty = None, False
        self.fig.clear()
        img = self.state.image
        if img is None:
            self.fig.text(0.5, 0.5, "Open an image first (button in the top bar)", ha="center", color="#777")
        else:
            try:
                self.draw(self.fig, img)
            except Exception as exc:
                self.fig.clear()
                self.fig.text(0.5, 0.5, f"Could not draw this view:\n{exc}", ha="center", color="#b00020")
        self.canvas.draw_idle()

    # --- small helpers for building controls
    def combo(self, frame, label, values, default, width=24):
        ttk.Label(frame, text=label).pack(side="left", padx=(8, 2))
        var = tk.StringVar(value=default)
        box = ttk.Combobox(frame, textvariable=var, values=list(values), width=width, state="readonly")
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", self.changed)
        return var

    def slider(self, frame, label, low, high, default, length=170, integer=True):
        ttk.Label(frame, text=label).pack(side="left", padx=(12, 2))
        var = tk.DoubleVar(value=default)
        value_label = ttk.Label(frame, width=5)

        def on_move(_=None):
            value_label.configure(text=str(int(var.get())) if integer else f"{var.get():.1f}")
            self.changed()
        ttk.Scale(frame, from_=low, to=high, variable=var, length=length, command=on_move).pack(side="left")
        value_label.pack(side="left")
        value_label.configure(text=str(default))
        return var

    def check(self, frame, label, default):
        var = tk.BooleanVar(value=default)
        ttk.Checkbutton(frame, text=label, variable=var, command=self.changed).pack(side="left", padx=8)
        return var


# ---------------------------------------------------------------------------- Lab 1
class BasicOpsPanel(Panel):
    title = "Basic operations"
    help_text = ("Lab 1: colour conversion, rotate / scale / shift / flip, add / subtract brightness, "
                 "bitwise operations. The slider is the 'p' value in the operation's name.")

    def build_controls(self, f):
        self.op = self.combo(f, "Operation:", basic_ops.OPERATIONS, "Rotate (p degrees)", 26)
        self.p = self.slider(f, "p =", 0, 255, 45)
        ttk.Button(f, text="Save result...", command=self.save).pack(side="right", padx=8)
        self.result = None

    def draw(self, fig, img):
        self.result = basic_ops.OPERATIONS[self.op.get()](img, self.p.get())
        a1, a2 = fig.subplots(1, 2)
        show(a1, img, "Original")
        show(a2, self.result, f"{self.op.get().split(' (')[0]}  -  {self.result.shape[1]} x {self.result.shape[0]} px")

    def save(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[
            ("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIFF", "*.tiff")])
        if path:
            basic_ops.save_as(self.result, path)


# ---------------------------------------------------------------------------- Lab 2
class FilterPanel(Panel):
    title = "Filters"
    help_text = ("Lab 2: correlation slides the kernel as is; convolution flips it first. "
                 "For a symmetric kernel (blur) they are identical, for Sobel/Emboss they are not.")

    def build_controls(self, f):
        self.kernel = self.combo(f, "Kernel:", spatial.KERNELS, "Sobel X (vertical edges)", 28)

    def draw(self, fig, img):
        gray = shrink(gray_of(img), 500)
        k = spatial.KERNELS[self.kernel.get()]
        _, corr = spatial.correlate2d(gray, k)
        _, conv = spatial.convolve2d(gray, k)
        diff = np.abs(corr.astype(np.int16) - conv.astype(np.int16)).astype(np.uint8)
        axes = fig.subplots(1, 4)
        show(axes[0], gray, "Original")
        show(axes[1], corr, "Correlation")
        show(axes[2], conv, "Convolution")
        show(axes[3], diff, "|Difference|")
        same = "symmetric kernel: both give the same result" if spatial.is_symmetric_kernel(k) \
            else "not symmetric: results differ"
        fig.suptitle(f"{self.kernel.get()} - {same}", fontsize=10)


# ---------------------------------------------------------------------------- Lab 3
class SamplingPanel(Panel):
    title = "Sampling & quantization"
    help_text = ("Lab 3: sampling lowers the spatial resolution, quantization lowers the number of grey levels. "
                 "'Detail' counts Canny edge pixels that survive.")

    def build_controls(self, f):
        self.factor = self.combo(f, "Keep 1 pixel in:", [1, 2, 4, 8, 16], "4", 6)
        self.levels = self.combo(f, "Grey levels:", [256, 32, 8, 4, 2], "8", 6)

    def draw(self, fig, img):
        gray = gray_of(img)
        factor, levels = int(self.factor.get()), int(self.levels.get())
        sampled = sampling.apply_sampling(gray, factor)
        quant = sampling.apply_quantization(gray, levels)
        both = sampling.apply_quantization(sampled, levels)
        axes = fig.subplots(1, 4)
        for ax, im, t in [(axes[0], gray, "Original"), (axes[1], sampled, f"Sampled 1/{factor}"),
                          (axes[2], quant, f"{levels} grey levels"), (axes[3], both, "Both")]:
            show(ax, im, f"{t}\ndetail {sampling.detail_score(im)}")


# ---------------------------------------------------------------------------- Lab 4
class ColorPanel(Panel):
    title = "Colour spaces"
    help_text = ("Lab 4: the same photo in different colour spaces, channel by channel, with histograms. "
                 "The 'colour signature' (average colour of the pill) is what the database stores.")

    def build_controls(self, f):
        self.space = self.combo(f, "Colour space:", color.COLOR_SPACES, "HSV", 12)

    def draw(self, fig, img):
        space = self.space.get()
        labels, channels = color.get_channels(img, space)
        n = len(channels)
        gs = fig.add_gridspec(2, 4, hspace=0.35)
        show(fig.add_subplot(gs[0, 0]), img, "Original")
        for i, (label, ch) in enumerate(zip(labels, channels)):
            show(fig.add_subplot(gs[0, i + 1]), ch, f"{space}: {label}")
            hist, lo, hi = color.channel_histogram(ch, space, i)
            ax = fig.add_subplot(gs[1, i + 1])
            ax.bar(np.arange(lo, hi), hist, width=1.0, color="steelblue")
            ax.set_title(f"{label} histogram", fontsize=8)
            ax.tick_params(labelsize=7)
        sig = color.color_signature(img, segmentation.segment_pill(img))
        ax = fig.add_subplot(gs[1, 0])
        swatch = np.zeros((40, 40, 3), np.uint8)
        swatch[:, :] = (int(sig["hue"]), int(sig["saturation"]), int(sig["value"]))
        ax.imshow(cv2.cvtColor(cv2.cvtColor(swatch, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pill colour: {sig['colorName']}", fontsize=9)
        ax.set_xlabel(f"H={sig['hue']} S={sig['saturation']} V={sig['value']}\nL={sig['L']} a={sig['a']} b={sig['b']}", fontsize=7)


# ---------------------------------------------------------------------------- Lab 5
class FourierPanel(Panel):
    title = "Fourier transform"
    help_text = ("Lab 5: the photo as frequencies. Low-pass keeps the smooth shape (blur); "
                 "high-pass keeps edges and fine speckle. D0 is the cut-off radius.")

    def build_controls(self, f):
        self.d0 = self.slider(f, "Cut-off D0:", 1, 120, 20, length=260)

    def draw(self, fig, img):
        gray = shrink(gray_of(img), 256)
        d0 = int(self.d0.get())
        fshift = frequency.compute_fshift(gray)
        low, low_mask = frequency.apply_ideal_filter(gray, d0, "low")
        high, _ = frequency.apply_ideal_filter(gray, d0, "high")
        axes = fig.subplots(2, 3).flatten()
        show(axes[0], gray, "Grey image")
        show(axes[1], frequency.magnitude_spectrum(fshift), "Magnitude spectrum (log)")
        show(axes[2], frequency.phase_spectrum(fshift), "Phase spectrum")
        show(axes[3], low_mask, f"Low-pass mask (D0={d0})")
        show(axes[4], low, "Low-pass result (blurred)")
        show(axes[5], high, "High-pass result (edges, detail)")


# ---------------------------------------------------------------------------- Lab 6
class EdgePanel(Panel):
    title = "Edge detection"
    help_text = ("Lab 6: Sobel, Canny and Laplacian-of-Gaussian on the same preprocessed image. "
                 "The bar chart counts edge pixels - lower thresholds give more edges (and more noise).")

    def build_controls(self, f):
        self.denoise = self.check(f, "Median denoise", True)
        self.clahe = self.check(f, "CLAHE", True)
        self.c_low = self.slider(f, "Canny low", 0, 300, 50, 110)
        self.c_high = self.slider(f, "high", 0, 500, 150, 110)
        self.sobel = self.slider(f, "Sobel >=", 1, 500, 100, 110)
        self.log = self.slider(f, "LoG >=", 1, 100, 10, 110)

    def draw(self, fig, img):
        gray = shrink(gray_of(img), 500)
        pre, results, counts = edges.run_all_methods(
            gray, self.denoise.get(), self.clahe.get(), int(self.c_low.get()), int(self.c_high.get()),
            self.sobel.get(), self.log.get())
        axes = fig.subplots(2, 3).flatten()
        show(axes[0], gray, "Original")
        show(axes[1], pre, "Preprocessed")
        for ax, name in zip(axes[2:5], ["Sobel", "Canny", "LoG"]):
            show(ax, 255 - results[name], f"{name} ({counts[name]} px)")
        axes[5].bar(list(counts), list(counts.values()), color=["steelblue", "seagreen", "indianred"])
        axes[5].set_title("Edge pixel count", fontsize=9)


# ---------------------------------------------------------------------------- Lab 7
class SegmentationPanel(Panel):
    title = "Segmentation"
    help_text = ("Lab 7: region growing starts at a seed pixel (click the left picture) and adds neighbours "
                 "whose grey value is close to the region's average. The two 'automatic' modes are what the "
                 "Identify tab uses to find the pill.")

    def build_controls(self, f):
        self.mode = self.combo(f, "Mode:", ["Region growing from clicked seed", "Automatic: threshold",
                                            "Automatic: region growing from corners"],
                               "Region growing from clicked seed", 34)
        self.thresh = self.slider(f, "Threshold:", 1, 80, 20, 150)
        self.conn = tk.IntVar(value=4)
        ttk.Label(f, text="Neighbours:").pack(side="left", padx=(12, 2))
        for n in (4, 8):
            ttk.Radiobutton(f, text=str(n), variable=self.conn, value=n, command=self.changed).pack(side="left")
        self.seed = None
        self._small = None
        self._ax_img = None
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def on_image_changed(self):
        self.seed = None
        super().on_image_changed()

    def on_click(self, event):
        if event.inaxes is self._ax_img and event.xdata is not None and self._small is not None:
            self.seed = (int(event.ydata), int(event.xdata))
            self.mode.set("Region growing from clicked seed")
            self.refresh()

    def draw(self, fig, img):
        small = shrink(img, 300)                       # region growing is plain Python: keep it small
        gray = gray_of(small)
        self._small = small
        h, w = gray.shape
        mode = self.mode.get()
        if mode.startswith("Region"):
            if self.seed is None:
                self.seed = (h // 2, w // 2)
            region, count, mean = segmentation.region_growing(gray, self.seed, self.thresh.get(), self.conn.get())
            label = f"Region growing: {count} px ({100.0 * count / (h * w):.1f}%), mean grey {mean:.0f}"
        else:
            method = "threshold" if "threshold" in mode else "region_growing"
            region = segmentation.segment_pill(small, method)
            label = f"Pill mask ({method}): {100.0 * region.mean():.1f}% of the picture"
        axes = fig.subplots(1, 3)
        self._ax_img = axes[0]
        show(axes[0], small, "Original (click to place seed)")
        if self.seed is not None and mode.startswith("Region"):
            axes[0].plot(self.seed[1], self.seed[0], "r+", markersize=14, markeredgewidth=2)
        show(axes[1], region.astype(np.uint8) * 255, "Region mask")
        show(axes[2], segmentation.make_overlay(gray, region), "Overlay", bgr=False)
        fig.suptitle(label, fontsize=10)


# ---------------------------------------------------------------------------- Lab 8
class FeaturePanel(Panel):
    title = "Features (SIFT / ORB / HOG)"
    help_text = ("Lab 8: SIFT (accurate, slower) vs ORB (fast, binary). The photo is rotated and matched back "
                 "to itself - good matches = rotation robustness. HOG is a global description built from edge directions.")

    def build_controls(self, f):
        self.angle = self.combo(f, "Rotate by:", [0, 15, 30, 45, 60, 90], "30", 6)
        ttk.Button(f, text="Run full rotation sweep", command=self.sweep_now).pack(side="left", padx=12)
        self.sweep = None

    def on_image_changed(self):
        self.sweep = None
        super().on_image_changed()

    @staticmethod
    def _match_view(g1, k1, d1, g2, k2, d2, norm):
        good = features.ratio_test_matches(d1, d2, norm)
        good = sorted(good, key=lambda m: m.distance)[:30]
        view = cv2.drawMatches(g1, k1, g2, k2, good, None, matchColor=(0, 200, 0),
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return view, len(features.ratio_test_matches(d1, d2, norm))

    def _detect(self, gray):
        t = time.perf_counter()
        ks, ds = features.detect_sift(gray)
        ts = time.perf_counter() - t
        t = time.perf_counter()
        ko, do = features.detect_orb(gray)
        to = time.perf_counter() - t
        return (ks, ds, ts), (ko, do, to)

    def draw(self, fig, img):
        gray = shrink(gray_of(img), 400)
        (ks, ds, ts), (ko, do, to) = self._detect(gray)
        angle = int(self.angle.get())
        rot = features.rotate_image(gray, angle)
        (ks2, ds2, _), (ko2, do2, _) = self._detect(rot)

        axes = fig.subplots(2, 3).flatten()
        show(axes[0], cv2.drawKeypoints(gray, ks, None, color=(0, 200, 255)), f"SIFT: {len(ks)} keypoints, {ts * 1000:.0f} ms", bgr=True)
        show(axes[1], cv2.drawKeypoints(gray, ko, None, color=(0, 255, 0)), f"ORB: {len(ko)} keypoints, {to * 1000:.0f} ms", bgr=True)
        hog_img = cv2.resize(gray, (128, 128))
        cell_hist = features.compute_hog(hog_img, 8, 9)
        show(axes[2], features.hog_visualization(hog_img, cell_hist, 8, 9), "HOG (global feature)")

        view, n = self._match_view(gray, ks, ds, rot, ks2, ds2, cv2.NORM_L2)
        show(axes[3], view, f"SIFT after {angle} deg: {n} good matches ({100.0 * n / max(len(ks), 1):.0f}%)")
        view, n = self._match_view(gray, ko, do, rot, ko2, do2, cv2.NORM_HAMMING)
        show(axes[4], view, f"ORB after {angle} deg: {n} good matches ({100.0 * n / max(len(ko), 1):.0f}%)")

        ax = axes[5]
        if self.sweep:
            ax.plot(self.sweep["angles"], self.sweep["sift"], "o-", label="SIFT")
            ax.plot(self.sweep["angles"], self.sweep["orb"], "s-", label="ORB")
            ax.set_xlabel("rotation (deg)", fontsize=8)
            ax.set_ylabel("good matches (%)", fontsize=8)
            ax.legend(fontsize=8)
            ax.set_title("Rotation robustness", fontsize=9)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "Press 'Run full rotation sweep'\nto compare SIFT and ORB\nat 0-90 degrees", ha="center", va="center", color="#777")

    def sweep_now(self):
        if self.state.image is None:
            return
        gray = shrink(gray_of(self.state.image), 400)
        (ks, ds, _), (ko, do, _) = self._detect(gray)
        angles = [0, 15, 30, 45, 60, 90]
        sift, orb = [], []
        for a in angles:
            rot = features.rotate_image(gray, a)
            (k2, d2, _), (o2, e2, _) = self._detect(rot)
            sift.append(100.0 * len(features.ratio_test_matches(ds, d2, cv2.NORM_L2)) / max(len(ks), 1))
            orb.append(100.0 * len(features.ratio_test_matches(do, e2, cv2.NORM_HAMMING)) / max(len(ko), 1))
        self.sweep = {"angles": angles, "sift": sift, "orb": orb}
        self.refresh()


ALL_PANELS = [BasicOpsPanel, FilterPanel, SamplingPanel, ColorPanel, FourierPanel, EdgePanel,
              SegmentationPanel, FeaturePanel]
