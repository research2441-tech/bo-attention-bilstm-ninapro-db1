"""Schematic figures for the NinaPro DB1 study (grayscale, no overlapping elements)."""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "font.size": 8,
                     "mathtext.fontset": "stix"})
G1, G2 = "#ececec", "#bdbdbd"


def box(ax, x, y, w, h, text, fill="white", lw=0.9, style="round,pad=0.02,rounding_size=0.06", fs=7.2, ls="-", it=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fill, ec="black", lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.25,
                style="italic" if it else "normal")
    return (x, y, w, h)


def arrow(ax, p, q, ls="-", lw=0.9, rad=0.0, head=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, lw=lw, ls=ls, color="black",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def fig_framework():
    fig, ax = plt.subplots(figsize=(7.1, 4.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
    W_, G_ = 14.2, 2.6
    xs = [0.6 + k * (W_ + G_) for k in range(6)]
    y1, h1 = 31, 13
    labels = ["DB1 sEMG\n10 electrodes\nRMS, 100 Hz",
              "Windowing\n200 ms, 100 ms\nstep, constant\nlabel",
              "Log transform\n+ z-score\n(statistics from\ntraining reps)",
              "Bi-LSTM\nencoder\n" + r"$\mathbf{h}_t\in\mathbb{R}^{2h}$",
              "Additive\nattention\n" + r"weights $\alpha_t$" + "\ncontext " + r"$\mathbf{c}$",
              "Softmax over\n53 classes\n" + r"$\hat{\mathbf{p}}=\mathrm{softmax}(\mathbf{z})$"]
    B = [box(ax, x, y1, W_, h1, t, fill=(G1 if k in (3, 4) else "white"), fs=6.8) for k, (x, t) in enumerate(zip(xs, labels))]
    for a, b in zip(B[:-1], B[1:]):
        arrow(ax, (a[0] + a[2], a[1] + a[3] / 2), (b[0], b[1] + b[3] / 2))
    # training objective above the encoder, attention and output
    L = box(ax, xs[3], 51.5, xs[5] + W_ - xs[3], 8.5,
            "Class-balanced cross-entropy " + r"$\mathcal{L}(\mathbf{z},y)$" + " with gain/noise augmentation\n(training only)", fs=6.8)
    arrow(ax, (xs[5] + W_ / 2, y1 + h1), (xs[5] + W_ / 2, 51.5))
    arrow(ax, (xs[3] + W_ / 2, 51.5), (xs[3] + W_ / 2, y1 + h1), ls=(0, (3, 2)))
    arrow(ax, (xs[4] + W_ / 2, 51.5), (xs[4] + W_ / 2, y1 + h1), ls=(0, (3, 2)))
    ax.text(xs[3] - 0.8, 48.0, r"$\partial\mathcal{L}/\partial\boldsymbol{\phi}$" + "\n(backprop)", ha="right", va="center", fontsize=6.6)
    # evaluation row under the output
    y2, h2 = 12, 11
    E1 = box(ax, xs[5], y2, W_, h2, "Test reps\n2, 5, 7\nper subject", fill=G2, fs=6.8)
    E2 = box(ax, xs[4], y2, W_, h2, "Faithfulness\n(deletion,\nsufficiency)", fs=6.8)
    E3 = box(ax, xs[3], y2, W_, h2, "Robustness\n(gain, noise,\nelectrode loss/shift)", fs=6.6)
    arrow(ax, (xs[5] + W_ / 2, y1), (xs[5] + W_ / 2, y2 + h2))
    arrow(ax, (E1[0], y2 + h2 / 2), (E2[0] + W_, y2 + h2 / 2))
    arrow(ax, (E2[0], y2 + h2 / 2), (E3[0] + W_, y2 + h2 / 2))
    # Bayesian optimisation loop on validation repetitions only
    px0, px1, py0, py1 = 0.6, 46.0, 1.0, 27.0
    box(ax, px0, py0, px1 - px0, py1 - py0, "", style="round,pad=0.02,rounding_size=0.08", ls="--", lw=0.8)
    ax.text((px0 + px1) / 2, py1 - 2.4, "Bayesian optimisation on validation repetitions only", ha="center", va="center",
            fontsize=6.9, style="italic")
    T1 = box(ax, 2.6, 11.5, 17, 10, "GP surrogate\n+ log-EI\n" + r"$\mathbf{u}\in[0,1]^{7}$", fs=6.8)
    T2 = box(ax, 27.0, 11.5, 17, 10, "Fitness\n" + r"$\mathcal{J}(\mathbf{u})$" + " = mean\nvalidation BA", fs=6.8)
    T3 = box(ax, 2.6, 2.8, 41.4, 6.4, "Panel subjects: fit reps 1, 3, 4, 6, 8; validate reps 9, 10", fs=6.5)
    arrow(ax, (T1[0] + T1[2], 18.8), (T2[0], 18.8))
    arrow(ax, (T2[0], 14.3), (T1[0] + T1[2], 14.3))
    ax.text(23.3, 19.6, "candidate", ha="center", va="bottom", fontsize=6.2)
    ax.text(23.3, 13.5, "score", ha="center", va="top", fontsize=6.2)
    arrow(ax, (35.5, T3[1] + T3[3]), (35.5, T2[1]))
    arrow(ax, (px1, py1 - 4), (xs[3] + 3.0, y1))
    ax.text(50.8, 28.4, r"selected $\boldsymbol{\lambda}^{\star}$", ha="center", va="center", fontsize=6.8,
            bbox=dict(fc="white", ec="none", pad=0.4))
    fig.savefig(os.path.join(OUT, "fig_framework.png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_protocol():
    fig, ax = plt.subplots(figsize=(3.45, 3.3))
    ax.set_xlim(0, 50); ax.set_ylim(0, 47); ax.axis("off")
    role_final = {1: "F", 3: "F", 4: "F", 6: "F", 8: "F", 9: "F", 10: "E", 2: "T", 5: "T", 7: "T"}
    role_inner = {1: "F", 3: "F", 4: "F", 6: "F", 8: "F", 9: "V", 10: "V", 2: "-", 5: "-", 7: "-"}
    fill = {"F": "white", "E": G1, "V": G1, "T": G2, "-": "white"}
    x0, w = 9.5, 3.9
    rows = [("Search", role_inner, 33.5), ("Final", role_final, 23.5)]
    ax.text(x0 + 5 * w, 44.5, "Repetition of every movement", ha="center", va="center", fontsize=6.9, style="italic")
    for k in range(10):
        ax.text(x0 + k * w + w / 2, 40.6, str(k + 1), ha="center", va="center", fontsize=6.9)
    for name, role, y in rows:
        ax.text(x0 - 1.0, y + 2.6, name, ha="right", va="center", fontsize=6.9)
        for k in range(10):
            r = role[k + 1]
            ax.add_patch(Rectangle((x0 + k * w, y), w, 5.2, fc=fill[r], ec="black", lw=0.7,
                                   hatch=("////" if r == "-" else None)))
            ax.text(x0 + k * w + w / 2, y + 2.6, "" if r == "-" else r, ha="center", va="center", fontsize=6.6)
    leg = [("F", "fit", "white"), ("E", "early stop", G1), ("V", "validation", G1), ("T", "test", G2)]
    for i, (c, lab, fc) in enumerate(leg):
        xx = 1.5 + i * 12.2
        ax.add_patch(Rectangle((xx, 15.2), 3.0, 3.0, fc=fc, ec="black", lw=0.6))
        ax.text(xx + 1.5, 16.7, c, ha="center", va="center", fontsize=6.0)
        ax.text(xx + 3.7, 16.7, lab, ha="left", va="center", fontsize=6.3)
    ax.add_patch(Rectangle((1.5, 10.3), 3.0, 3.0, fc="white", ec="black", lw=0.6, hatch="////"))
    ax.text(5.2, 11.8, "never used in the search", ha="left", va="center", fontsize=6.3)
    box(ax, 1.5, 0.6, 47, 7.6, "Leave-one-subject-out: fit on 26 subjects, test on the held-out\nsubject (normalised with its own repetition-1 recording)", fs=6.4)
    fig.savefig(os.path.join(OUT, "fig_protocol.png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_framework(); fig_protocol()
