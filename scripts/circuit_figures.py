"""
Phase 3 step 4: circuit and pipeline diagrams that match the exact v3 model.

  fig_pipeline.pdf   v3 pipeline (pre-biopsy features, grouped folds, PCA-12, 8q tree)
  fig_blocks.pdf     three panels: (a) angle encoding with re-upload (RY x1..x8 on q0..q7,
                     RZ x9..x12 on q0..q3); (b) conv block RY(t1) RY(t2), CNOT a->b, RY(t3) RY(t4),
                     CNOT b->a, RZ(t5) RZ(t6); (c) pooling block CRZ(phi1), CRX(phi2), source -> sink
  fig_encoding.pdf, fig_conv.pdf, fig_pool.pdf   the same three panels as separate files
  fig_tree_8q.pdf    full 8-qubit tree (3 conv layers per level, pooling 8 -> 4 -> 2 -> 1)

Figures carry no titles (captions are in the paper); multi-panel figures use (a), (b), (c).

The tree layout, encoding rounds and configuration values are read from v3core
(qcnn_plan, champion.yaml) so the drawings follow the model. All text >= 8 pt, vector PDF.

Usage:
  python scripts/circuit_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

from v3core.config import load_config  # noqa: E402
from v3core.hqcnn import qcnn_plan  # noqa: E402
from v3core.plotting import FIG_DIR, save_pdf, setup_style  # noqa: E402

GATE_FC = {"RY": "#dbe9f6", "RZ": "#fde2cf", "RX": "#e3f1d8", "CTRL": "#f3e1f5"}
FS = 9
CONV_LABEL = r"$U_{\mathrm{conv}}^{\times %d}$"


class Circuit:
    """Minimal vector circuit drawer: wires top-to-bottom, time left-to-right."""

    def __init__(self, ax, n_wires, labels, x0=0.0, dy=1.0):
        self.ax, self.n, self.dy, self.x0 = ax, n_wires, dy, x0
        self.labels = labels

    def y(self, w):
        return -w * self.dy

    def gate(self, x, w, text, kind="RY", width=0.9):
        self.ax.add_patch(Rectangle((x - width / 2, self.y(w) - 0.3), width, 0.6,
                                    fc=GATE_FC.get(kind, "white"), ec="black", lw=0.8, zorder=3))
        self.ax.text(x, self.y(w), text, ha="center", va="center", fontsize=FS, zorder=4)

    def cnot(self, x, ctrl, tgt):
        self.ax.plot([x, x], [self.y(ctrl), self.y(tgt)], color="black", lw=0.9, zorder=2)
        self.ax.add_patch(Circle((x, self.y(ctrl)), 0.08, color="black", zorder=4))
        self.ax.add_patch(Circle((x, self.y(tgt)), 0.18, fc="white", ec="black", lw=0.9, zorder=4))
        self.ax.plot([x - 0.18, x + 0.18], [self.y(tgt)] * 2, color="black", lw=0.9, zorder=5)
        self.ax.plot([x, x], [self.y(tgt) - 0.18, self.y(tgt) + 0.18], color="black", lw=0.9, zorder=5)

    def controlled(self, x, ctrl, tgt, text, width=1.0):
        self.ax.plot([x, x], [self.y(ctrl), self.y(tgt)], color="black", lw=0.9, zorder=2)
        self.ax.add_patch(Circle((x, self.y(ctrl)), 0.08, color="black", zorder=4))
        self.gate(x, tgt, text, kind="CTRL", width=width)

    def block(self, x, a, b, text, width=1.1):
        """Two-qubit block on wires a < b; drawn only on those wires if not adjacent."""
        if b - a == 1:
            y0, y1 = self.y(b) - 0.3, self.y(a) + 0.3
            self.ax.add_patch(FancyBboxPatch((x - width / 2, y0), width, y1 - y0, boxstyle="round,pad=0.02",
                                             fc=GATE_FC["RY"], ec="black", lw=0.8, zorder=3))
            self.ax.text(x, (y0 + y1) / 2, text, ha="center", va="center", fontsize=FS, zorder=4)
            return
        self.ax.plot([x, x], [self.y(a), self.y(b)], color="black", lw=2.4, zorder=2)
        self.ax.plot([x, x], [self.y(a), self.y(b)], color=GATE_FC["RY"], lw=1.2, zorder=2)
        for w in (a, b):
            self.ax.add_patch(FancyBboxPatch((x - width / 2, self.y(w) - 0.3), width, 0.6,
                                             boxstyle="round,pad=0.02", fc=GATE_FC["RY"], ec="black",
                                             lw=0.8, zorder=3))
            self.ax.text(x, self.y(w), text, ha="center", va="center", fontsize=FS, zorder=4)

    def wires(self, x_end, stop=None):
        stop = stop or {}
        for w in range(self.n):
            xe = stop.get(w, x_end)
            self.ax.plot([self.x0 - 0.3, xe], [self.y(w)] * 2, color="black", lw=0.8, zorder=1)
            if w in stop:
                self.ax.plot([xe - 0.08, xe + 0.08], [self.y(w) - 0.12, self.y(w) + 0.12], color="0.4", lw=0.8)
            self.ax.text(self.x0 - 0.45, self.y(w), self.labels[w], ha="right", va="center", fontsize=FS)

    def finish(self, x_end, margin_top=0.8, margin_bottom=0.8, left=1.4):
        self.ax.set_xlim(self.x0 - left, x_end + 0.4)
        self.ax.set_ylim(self.y(self.n - 1) - margin_bottom, margin_top)
        self.ax.set_aspect("equal")
        self.ax.axis("off")


def panel_label(ax, text) -> None:
    ax.text(0.0, 1.0, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS, weight="bold")


def draw_encoding(ax, cfg) -> None:
    n, nf = int(cfg["hqcnn"]["n_qubits"]), int(cfg["preprocessing"]["pca_dim"])
    gates = list(cfg["hqcnn"]["reupload_gates"])
    c = Circuit(ax, n, [f"$q_{i}$" for i in range(n)])
    x = 0.6
    for r, g in enumerate(gates):
        start = r * n
        if start >= nf:
            break
        for q in range(n):
            if start + q < nf:
                label = r"$R_{%s}(x_{%d})$" % (g[1].lower(), start + q + 1)
                c.gate(x, q, label, kind=g, width=1.5)
        x += 1.75
    c.wires(x - 0.4)
    c.finish(x - 0.4)


def draw_conv(ax) -> None:
    c = Circuit(ax, 2, ["$a$", "$b$"], dy=1.0)
    c.gate(0.6, 0, r"$R_y(\theta_1)$", "RY", 1.1)
    c.gate(0.6, 1, r"$R_y(\theta_2)$", "RY", 1.1)
    c.cnot(1.6, 0, 1)
    c.gate(2.6, 0, r"$R_y(\theta_3)$", "RY", 1.1)
    c.gate(2.6, 1, r"$R_y(\theta_4)$", "RY", 1.1)
    c.cnot(3.6, 1, 0)
    c.gate(4.6, 0, r"$R_z(\theta_5)$", "RZ", 1.1)
    c.gate(4.6, 1, r"$R_z(\theta_6)$", "RZ", 1.1)
    c.wires(5.4)
    c.finish(5.4, margin_top=0.6, margin_bottom=0.5)


def draw_pool(ax) -> None:
    c = Circuit(ax, 2, ["source $a$", "sink $b$"], dy=1.0)
    c.controlled(0.7, 0, 1, r"$R_z(\phi_1)$", 1.1)
    c.controlled(2.0, 0, 1, r"$R_x(\phi_2)$", 1.1)
    c.wires(2.8, stop={0: 2.6})
    ax.text(2.75, c.y(0), "discarded", ha="left", va="center", fontsize=8, color="0.3")
    c.finish(3.9, margin_top=0.6, margin_bottom=0.5, left=1.9)


def fig_blocks(cfg) -> None:
    """Encoding, convolution and pooling blocks as one three-panel figure, plus single panels."""
    fig = plt.figure(figsize=(7.0, 3.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1, 1], wspace=0.08, hspace=0.35)
    ax_a, ax_b, ax_c = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    draw_encoding(ax_a, cfg)
    draw_conv(ax_b)
    draw_pool(ax_c)
    for ax, t in ((ax_a, "(a)"), (ax_b, "(b)"), (ax_c, "(c)")):
        panel_label(ax, t)
    save_pdf(fig, FIG_DIR / "fig_blocks.pdf")
    for name, size, draw in (("fig_encoding", (3.4, 3.4), lambda ax: draw_encoding(ax, cfg)),
                             ("fig_conv", (3.4, 1.35), draw_conv), ("fig_pool", (2.9, 1.35), draw_pool)):
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        save_pdf(fig, FIG_DIR / f"{name}.pdf")


def fig_tree(cfg) -> None:
    n, cl = int(cfg["hqcnn"]["n_qubits"]), int(cfg["hqcnn"]["conv_layers"])
    ops, ro = qcnn_plan(n, cl, cfg["hqcnn"]["pooling_survivor"], "tree")
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    c = Circuit(ax, n, [f"$q_{i}$" for i in range(n)], dy=0.8)
    x, stop = 0.7, {}
    for q in range(n):
        c.gate(x, q, "Enc", "RY", 0.9)
    x += 1.2
    level_ops, cur, prev = [], [], None
    for t, p in ops:
        if prev == "pool" and t == "conv":
            level_ops.append(cur)
            cur = []
        cur.append((t, p))
        prev = t
    level_ops.append(cur)
    for lops in level_ops:
        pairs = list(dict.fromkeys(p for t, p in lops if t == "conv"))
        for a, b in pairs:
            c.block(x, a, b, CONV_LABEL % cl)
        x += 1.3
        for a, b in (p for t, p in lops if t == "pool"):
            c.controlled(x, a, b, "P", 0.6)
            stop[a] = x + 0.35
        x += 1.1
    c.gate(x, ro, r"$\langle Z\rangle$", kind="M", width=0.8)
    c.wires(x - 0.4, stop=stop)
    c.finish(x + 0.2, margin_top=0.5, margin_bottom=0.5)
    save_pdf(fig, FIG_DIR / "fig_tree_8q.pdf")


def fig_pipeline(cfg) -> None:
    """Two rows, orthogonal connectors: data -> radiomics / clinical -> fusion, then angle scaling -> HQCNN -> read-out.

    Coordinates are in inches (the axes fill the figure), so box sizes and gaps are the printed sizes.
    """
    pp, h = cfg["preprocessing"], cfg["hqcnn"]
    res = {"borderline": "Borderline-SMOTE", "smote": "SMOTE", "none": "no resampling"}[pp["resampler"]]
    data_fc, prep_fc, q_fc, ec = "#eef1f5", "#dbe9f6", "#e9dff5", "#333333"
    FW, FH = 6.8, 3.3
    y1, y2 = 2.3, 0.45                      # row centres
    boxes = {   # key: (x_left, y_centre, width, height, title, body, colour)
        "data": (0.05, y1, 1.35, 1.0, "PROSTATEx", "327 lesions\n204 patients\n76 csPCa", data_fc),
        "rad": (1.95, y1 + 0.5, 1.95, 0.84, "Radiomics",
                "352 features (T2W, ADC,\nDWI, K$^{\\mathrm{trans}}$); z-score,\n"
                f"best {pp['k']} by ANOVA $F$", prep_fc),
        "clin": (1.95, y1 - 0.5, 1.95, 0.84, "Clinical (pre-biopsy)",
                 "PSA, PSAD, volume, age;\nmedian impute, z-score;\nno Gleason grade", prep_fc),
        "pca": (4.5, y1, 2.25, 1.0, "Fusion and PCA",
                f"concatenate,\n{pp['pca_dim']} principal components,\n{res}", prep_fc),
        "scale": (0.05, y2, 1.8, 0.84, "Angle scaling", "min-max to $[0,\\pi]$\n(training-fold range)", prep_fc),
        "hqcnn": (2.35, y2, 2.1, 0.84, "Tree HQCNN",
                  f"{h['n_qubits']} qubits, RY/RZ re-upload,\n{h['conv_layers']} conv layers per level,\n"
                  "CRZ+CRX pooling", q_fc),
        "out": (4.95, y2, 1.8, 0.84, "Read-out",
                "$a\\langle Z_7\\rangle + b$, sigmoid,\nF1-optimal threshold", q_fc),
    }
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis("off")
    patches, texts = {}, {}
    for k, (x, y, w, hh, title, body, fc) in boxes.items():
        patches[k] = ax.add_patch(FancyBboxPatch((x, y - hh / 2), w, hh, boxstyle="round,pad=0,rounding_size=0.06",
                                                 fc=fc, ec=ec, lw=0.8))
        top = y + hh / 2
        texts[k] = [ax.text(x + w / 2, top - 0.08, title, ha="center", va="top", fontsize=FS, weight="bold"),
                    ax.text(x + w / 2, top - 0.29, body, ha="center", va="top", fontsize=8, linespacing=1.25)]

    line = dict(color=ec, lw=0.8, solid_capstyle="butt")
    arrow = dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=0.8, color=ec, shrinkA=0, shrinkB=0)

    def L(k):
        return boxes[k][0]

    def R(k):
        return boxes[k][0] + boxes[k][2]

    def arrow_to(xy_from, xy_to):
        ax.annotate("", xy=xy_to, xytext=xy_from, arrowprops=arrow)

    # split: PROSTATEx -> vertical bus -> radiomics and clinical
    ys_r, ys_c = boxes["rad"][1], boxes["clin"][1]
    xs = (R("data") + L("rad")) / 2
    ax.plot([R("data"), xs], [y1, y1], **line)
    ax.plot([xs, xs], [ys_c, ys_r], **line)
    arrow_to((xs, ys_r), (L("rad"), ys_r))
    arrow_to((xs, ys_c), (L("clin"), ys_c))
    # merge: radiomics and clinical -> vertical bus -> fusion
    xm = (R("rad") + L("pca")) / 2
    ax.plot([R("rad"), xm], [ys_r, ys_r], **line)
    ax.plot([R("clin"), xm], [ys_c, ys_c], **line)
    ax.plot([xm, xm], [ys_c, ys_r], **line)
    arrow_to((xm, y1), (L("pca"), y1))
    # row 1 -> row 2: down from fusion, left through the gap between rows, down into angle scaling
    xp = L("pca") + boxes["pca"][2] / 2
    xa = L("scale") + boxes["scale"][2] / 2
    y_top2 = y2 + boxes["scale"][3] / 2
    y_bot1 = min(boxes[k][1] - boxes[k][3] / 2 for k in ("data", "clin", "pca"))
    yg = (y_top2 + y_bot1) / 2
    ax.plot([xp, xp, xa], [y1 - boxes["pca"][3] / 2, yg, yg], **line)
    arrow_to((xa, yg), (xa, y_top2))
    # row 2
    arrow_to((R("scale"), y2), (L("hqcnn"), y2))
    arrow_to((R("hqcnn"), y2), (L("out"), y2))

    # every box must lie inside the figure, and every text inside its box with a small margin
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    for k, (x, y, w, hh, *_) in boxes.items():
        if not (0 <= x and x + w <= FW and 0 <= y - hh / 2 and y + hh / 2 <= FH):
            raise ValueError(f"box {k!r} extends past the figure")
    for k, p in patches.items():
        bb = p.get_window_extent(rnd).padded(-0.04 * fig.dpi)
        for t in texts[k]:
            tb = t.get_window_extent(rnd)
            if not (bb.x0 <= tb.x0 and tb.x1 <= bb.x1 and bb.y0 <= tb.y0 and tb.y1 <= bb.y1):
                raise ValueError(f"text overflows box {k!r}: {t.get_text()!r}")
    save_pdf(fig, FIG_DIR / "fig_pipeline.pdf")


def main() -> None:
    setup_style()
    cfg = load_config()
    fig_pipeline(cfg)
    fig_blocks(cfg)
    fig_tree(cfg)
    print(f"  wrote fig_pipeline, fig_blocks (+ fig_encoding, fig_conv, fig_pool), fig_tree_8q -> {FIG_DIR}")


if __name__ == "__main__":
    main()
