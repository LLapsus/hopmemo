import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

import numpy as np

from hopfield import HopfieldNetwork


def _parse_float(s: str, default: float) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _parse_int(s: str, default: int) -> int:
    try:
        return int(s)
    except Exception:
        return default


class PatternGrid(ttk.Frame):
    def __init__(self, master, side: int, cell: int = 20, clickable: bool = True, title: str = ""):
        super().__init__(master)
        self.side = int(side)
        self.cell = int(cell)
        self.clickable = bool(clickable)

        if title:
            ttk.Label(self, text=title).pack(anchor="w")

        w = self.side * self.cell
        h = self.side * self.cell
        self.canvas = tk.Canvas(self, width=w, height=h, bg="white", highlightthickness=1, highlightbackground="#999")
        self.canvas.pack()

        self._rects = np.empty((self.side, self.side), dtype=int)
        self.state = -np.ones((self.side, self.side), dtype=int)

        for r in range(self.side):
            for c in range(self.side):
                x0, y0 = c * self.cell, r * self.cell
                x1, y1 = x0 + self.cell, y0 + self.cell
                rid = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ddd", fill="white")
                self._rects[r, c] = rid

        if self.clickable:
            self.canvas.bind("<Button-1>", self._on_click)

        self._redraw_all()

    def _color(self, v: int) -> str:
        return "black" if v == 1 else "white"

    def _redraw_cell(self, r: int, c: int) -> None:
        rid = int(self._rects[r, c])
        self.canvas.itemconfig(rid, fill=self._color(int(self.state[r, c])))

    def _redraw_all(self) -> None:
        for r in range(self.side):
            for c in range(self.side):
                self._redraw_cell(r, c)

    def _on_click(self, event):
        c = event.x // self.cell
        r = event.y // self.cell
        if 0 <= r < self.side and 0 <= c < self.side:
            self.state[r, c] *= -1
            self._redraw_cell(r, c)

    def clear(self):
        self.state.fill(-1)
        self._redraw_all()

    def invert(self):
        self.state *= -1
        self._redraw_all()

    def add_noise(self, p: float):
        p = float(np.clip(p, 0.0, 1.0))
        n = self.side * self.side
        k = int(round(p * n))
        if k <= 0:
            return
        idx = np.random.choice(n, size=k, replace=False)
        flat = self.state.reshape(-1)
        flat[idx] *= -1
        self._redraw_all()

    def set_vector(self, vec):
        vec = np.asarray(vec, dtype=int).reshape(self.side * self.side)
        if not np.isin(vec, (-1, 1)).all():
            raise ValueError("pattern must be {+1,-1}")
        self.state = vec.reshape(self.side, self.side).copy()
        self._redraw_all()

    def get_vector(self):
        return self.state.reshape(-1).copy()


class HopfieldApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hopfield Network (basic)")
        self.resizable(False, False)

        self.side_var = tk.StringVar(value="8")
        self.method_var = tk.StringVar(value="hebbian")
        self.theta_var = tk.StringVar(value="0.0")
        self.iters_var = tk.StringVar(value="50")
        self.rule_var = tk.StringVar(value="async")
        self.local_bias_var = tk.BooleanVar(value=False)
        self.noise_var = tk.StringVar(value="0.10")

        self.hop = None
        self._build_ui()
        self._recreate_network()

    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        top = ttk.Frame(root)
        top.pack(fill="x")

        ttk.Label(top, text="Side").pack(side="left")
        side_box = ttk.Combobox(top, textvariable=self.side_var, width=5, values=["8", "10", "14", "16", "28"], state="readonly")
        side_box.pack(side="left", padx=(6, 14))
        side_box.bind("<<ComboboxSelected>>", lambda e: self._recreate_network())

        ttk.Label(top, text="Learning").pack(side="left")
        method_box = ttk.Combobox(
            top,
            textvariable=self.method_var,
            width=14,
            values=["hebbian", "storkey", "pinv_centered", "pinv_damped"],
            state="readonly",
        )
        method_box.pack(side="left", padx=(6, 14))
        method_box.bind("<<ComboboxSelected>>", lambda e: self._apply_learning_method())

        ttk.Label(top, text="theta").pack(side="left")
        ttk.Entry(top, textvariable=self.theta_var, width=7).pack(side="left", padx=(6, 14))

        ttk.Label(top, text="max_iter").pack(side="left")
        ttk.Entry(top, textvariable=self.iters_var, width=7).pack(side="left", padx=(6, 14))

        ttk.Checkbutton(top, text="local_biases", variable=self.local_bias_var).pack(side="left")

        rule = ttk.Frame(root)
        rule.pack(fill="x", pady=(8, 0))
        ttk.Label(rule, text="Update rule:").pack(side="left")
        ttk.Radiobutton(rule, text="async", value="async", variable=self.rule_var).pack(side="left", padx=6)
        ttk.Radiobutton(rule, text="sync", value="sync", variable=self.rule_var).pack(side="left", padx=6)

        mid = ttk.Frame(root)
        mid.pack(fill="x", pady=(10, 0))

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(mid, textvariable=self.status).pack(side="left")

        body = ttk.Frame(root)
        body.pack(pady=(10, 0))

        self.grid_in = PatternGrid(body, side=8, cell=24, clickable=True, title="Input")
        self.grid_in.grid(row=0, column=0, padx=(0, 10))

        self.grid_out = PatternGrid(body, side=8, cell=24, clickable=False, title="Output")
        self.grid_out.grid(row=0, column=1, padx=(10, 0))

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(12, 0))

        ttk.Button(controls, text="Clear", command=self._clear).pack(side="left")
        ttk.Button(controls, text="Invert", command=self._invert).pack(side="left", padx=6)

        ttk.Label(controls, text="Noise p").pack(side="left", padx=(14, 0))
        ttk.Entry(controls, textvariable=self.noise_var, width=6).pack(side="left", padx=6)
        ttk.Button(controls, text="Apply Noise", command=self._noise).pack(side="left")

        ttk.Separator(controls, orient="vertical").pack(side="left", fill="y", padx=12)

        ttk.Button(controls, text="Memorize", command=self._memorize).pack(side="left")
        ttk.Button(controls, text="Retrieve", command=self._retrieve).pack(side="left", padx=6)
        ttk.Button(controls, text="Reset Network", command=self._reset_network).pack(side="left", padx=6)

        mem = ttk.Frame(root)
        mem.pack(fill="x", pady=(12, 0))
        ttk.Label(mem, text="Memories:").pack(anchor="w")
        self.mem_list = tk.Listbox(mem, height=4)
        self.mem_list.pack(fill="x")
        self.mem_list.bind("<<ListboxSelect>>", self._on_select_memory)

    def _recreate_network(self):
        side = _parse_int(self.side_var.get(), 8)
        side = max(2, side)
        n = side * side

        self.grid_in.destroy()
        self.grid_out.destroy()

        body = self.winfo_children()[0].winfo_children()[3]  # root -> body frame
        self.grid_in = PatternGrid(body, side=side, cell=max(10, 240 // side), clickable=True, title="Input")
        self.grid_in.grid(row=0, column=0, padx=(0, 10))
        self.grid_out = PatternGrid(body, side=side, cell=max(10, 240 // side), clickable=False, title="Output")
        self.grid_out.grid(row=0, column=1, padx=(10, 0))

        self.hop = HopfieldNetwork(n_neurons=n, learning_method=self.method_var.get())
        self.mem_list.delete(0, tk.END)
        self.status.set(f"Network: {n} neurons ({side}x{side}).")

    def _apply_learning_method(self):
        if self.hop is None:
            return
        self.hop.learning_method = self.method_var.get()
        self.status.set(f"Learning method set to {self.hop.learning_method} (weights recomputed on next memorize).")

    def _reset_network(self):
        if self.hop is None:
            return
        self.hop.reset_network()
        self.mem_list.delete(0, tk.END)
        self.status.set("Network reset (weights + memories cleared).")

    def _clear(self):
        self.grid_in.clear()
        self.grid_out.clear()

    def _invert(self):
        self.grid_in.invert()

    def _noise(self):
        p = _parse_float(self.noise_var.get(), 0.10)
        self.grid_in.add_noise(p)

    def _memorize(self):
        if self.hop is None:
            return
        vec = self.grid_in.get_vector()
        label = simpledialog.askstring("Memorize", "Label (optional):", parent=self)
        if not label:
            label = f"mem_{self.hop.num_memories() + 1}"
        try:
            self.hop.memorize(vec, labels=[label])
        except Exception as e:
            messagebox.showerror("Memorize failed", str(e))
            return
        self.mem_list.insert(tk.END, label)
        self.status.set(f"Stored: {label}. Total memories: {self.hop.num_memories()}")

    def _retrieve(self):
        if self.hop is None:
            return
        vec = self.grid_in.get_vector()
        theta = _parse_float(self.theta_var.get(), 0.0)
        iters = _parse_int(self.iters_var.get(), 50)
        rule = self.rule_var.get()
        use_local = bool(self.local_bias_var.get())
        try:
            out = self.hop.retrieve(vec, theta=theta, max_iterations=iters, update_rule=rule, use_local_biases=use_local)
        except Exception as e:
            messagebox.showerror("Retrieve failed", str(e))
            return
        self.grid_out.set_vector(out)
        self.status.set(f"Retrieved (method={self.hop.learning_method}, rule={rule}, local_biases={use_local}).")

    def _on_select_memory(self, _event):
        if self.hop is None or self.hop.num_memories() == 0:
            return
        sel = self.mem_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        try:
            pat = self.hop.memories[idx]
        except Exception:
            return
        self.grid_in.set_vector(pat)
        self.status.set(f"Loaded memory #{idx + 1} into input.")

if __name__ == "__main__":
    app = HopfieldApp()
    app.mainloop()
