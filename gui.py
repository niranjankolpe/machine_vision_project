"""
gui.py
======
Simple Tkinter GUI.
Upload image -> detect mango -> show result.
That's it.

Run: python gui.py
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

# ── Layout ────────────────────────────────────────────────────────────
WIN_W  = 620
WIN_H  = 560
IMG_W  = 260
IMG_H  = 260
BG     = "#1e1e2e"
PANEL  = "#2a2a3e"
GREEN  = "#a6e3a1"
RED    = "#f38ba8"
YELLOW = "#f9e2af"
FG     = "#cdd6f4"
GREY   = "#6c7086"


def bgr_to_tk(bgr, w=IMG_W, h=IMG_H):
    """Convert BGR numpy array to Tkinter-displayable image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (30, 30, 46))
    x = (w - pil.width)  // 2
    y = (h - pil.height) // 2
    canvas.paste(pil, (x, y))
    return ImageTk.PhotoImage(canvas)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Mango Harvest Readiness")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self._build()

    def _build(self):
        # Title
        tk.Label(self.root,
                 text="Mango Harvest Readiness",
                 font=("Helvetica", 14, "bold"),
                 bg=BG, fg=YELLOW
                 ).place(x=0, y=10, width=WIN_W)

        # ── Image panels ──────────────────────────────────────────────
        blank = ImageTk.PhotoImage(
            Image.new("RGB", (IMG_W, IMG_H), (42, 42, 62)))

        # Left — original with bbox
        tk.Label(self.root, text="Detected Image",
                 font=("Helvetica", 9), bg=BG, fg=FG
                 ).place(x=30, y=46, width=IMG_W)
        self.left_lbl = tk.Label(self.root, bg=PANEL)
        self.left_lbl.place(x=30, y=64, width=IMG_W, height=IMG_H)
        self.left_lbl.configure(image=blank)
        self.left_lbl.image = blank

        # Right — crop
        tk.Label(self.root, text="Mango Crop (ROI)",
                 font=("Helvetica", 9), bg=BG, fg=FG
                 ).place(x=330, y=46, width=IMG_W)
        self.right_lbl = tk.Label(self.root, bg=PANEL)
        self.right_lbl.place(x=330, y=64, width=IMG_W, height=IMG_H)
        self.right_lbl.configure(image=blank)
        self.right_lbl.image = blank

        self._blank = blank

        # ── Result box ────────────────────────────────────────────────
        tk.Frame(self.root, bg=PANEL, width=WIN_W-40, height=90
                 ).place(x=20, y=340)

        self.label_var = tk.StringVar(value="—")
        self.info_var  = tk.StringVar(value="Upload a mango image to begin")

        self.label_lbl = tk.Label(self.root,
                                   textvariable=self.label_var,
                                   font=("Helvetica", 16, "bold"),
                                   bg=PANEL, fg=YELLOW)
        self.label_lbl.place(x=20, y=355, width=WIN_W-40)

        tk.Label(self.root,
                 textvariable=self.info_var,
                 font=("Helvetica", 10),
                 bg=PANEL, fg=FG
                 ).place(x=20, y=392, width=WIN_W-40)

        # ── Status ─────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="")
        tk.Label(self.root,
                 textvariable=self.status_var,
                 font=("Helvetica", 8),
                 bg=BG, fg=GREY
                 ).place(x=0, y=446, width=WIN_W)

        # ── Buttons ────────────────────────────────────────────────────
        tk.Button(self.root, text="Upload Image",
                  command=self._upload,
                  font=("Helvetica", 11), bg="#313244", fg=FG,
                  relief="flat", padx=16, pady=7, cursor="hand2"
                  ).place(x=150, y=468)

        tk.Button(self.root, text="Quit",
                  command=self.root.quit,
                  font=("Helvetica", 11), bg="#313244", fg=GREY,
                  relief="flat", padx=16, pady=7, cursor="hand2"
                  ).place(x=360, y=468)

    def _set(self, lbl, bgr):
        tk_img = bgr_to_tk(bgr)
        lbl.configure(image=tk_img)
        lbl.image = tk_img

    def _upload(self):
        path = filedialog.askopenfilename(
            title="Select mango image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        self.status_var.set("Detecting... please wait.")
        self.root.update()
        try:
            self._run(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            raise

    def _run(self, path):
        from detector import detect_and_classify
        r = detect_and_classify(path)

        # Show annotated image
        self._set(self.left_lbl, r['result_img'])

        # Show first crop (or full image if no detection)
        self._set(self.right_lbl, r['detections'][0]['crop'])

        # Result label
        label = r['detections'][0]['label']
        conf  = r['detections'][0]['confidence']
        n     = len(r['detections'])

        color = GREEN if label == "Harvest Ready" else RED
        self.label_var.set(label)
        self.label_lbl.configure(fg=color)

        if conf:
            info = f"YOLO confidence: {conf*100:.0f}%"
        else:
            info = "No mango detected — classified from full image"

        if n > 1:
            info += f"  |  {n} mangoes found"

        self.info_var.set(info)
        self.status_var.set(f"Done — {os.path.basename(path)}")

        # Save predicted image with bbox
        import datetime
        label_safe = r['detections'][0]['label'].replace(" ", "_").replace("(", "").replace(")", "")
        conf_val   = r['detections'][0]['confidence']
        conf_str   = f"{conf_val*100:.0f}" if conf_val else "NA"
        timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename   = f"{label_safe}_{conf_str}_{timestamp}.jpg"
        os.makedirs("predictions", exist_ok=True)
        cv2.imwrite(os.path.join("predictions", filename), r['result_img'])
        self.status_var.set(f"Done — {os.path.basename(path)}  |  Saved: {filename}")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
