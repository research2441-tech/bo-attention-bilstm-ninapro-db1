"""Result figures (grayscale) built from results.json and the stored runs.

usage: python fig_results.py RESULTS.json RUNDIR DATADIR OUTDIR
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import softmax, load_subject, Scaler, mask, TRAIN_REPS

R = json.load(open(sys.argv[1])); RUN, DATA, OUT = sys.argv[2], sys.argv[3], sys.argv[4]
os.environ["DB1"] = DATA
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "font.size": 8,
                     "mathtext.fontset": "stix", "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "legend.frameon": False})
NAME = {"LDA": "LDA", "SVM": "SVM", "RF": "RF", "TCN_DEF": "TCN", "BILSTM_DEF": "Bi-LSTM", "ATTN_DEF": "Att-BiLSTM\n(manual)",
        "ATTN_RS": "Att-BiLSTM\n(RS)", "ATTN_BO": "Att-BiLSTM\n(BO)"}
ORDER = ["LDA", "SVM", "RF", "TCN_DEF", "BILSTM_DEF", "ATTN_DEF", "ATTN_RS", "ATTN_BO"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_signals():
    import scipy.io  # noqa: F401  (only numpy files are read here)
    X, y, r = load_subject(1)
    sc = Scaler().fit(X[mask(r, TRAIN_REPS)])
    # reconstruct a continuous stretch from consecutive windows of repetition 1 for two movements
    fig = plt.figure(figsize=(3.45, 2.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.55, wspace=0.35)
    for row, cls in enumerate((1, 9)):
        idx = np.where((y == cls) & (r == 1))[0]
        seq = np.concatenate([X[i][:10] for i in idx])          # 10 new samples per window (step 10)
        ax = fig.add_subplot(gs[row, 0])
        t = np.arange(len(seq)) / 100
        for c, ls in zip((0, 2, 4, 8), ("-", "--", ":", "-.")):
            ax.plot(t, seq[:, c], color="black", lw=0.8, ls=ls, label=f"ch {c + 1}")
        ax.set_xlabel("time in movement (s)", labelpad=1); ax.set_ylabel("envelope (a.u.)", labelpad=1)
        ax.set_title(f"movement A{cls}", fontsize=7.5, pad=2)
        if row == 0:
            ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.8, columnspacing=0.8)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
        axh = fig.add_subplot(gs[row, 1])
        win = sc(X[idx[len(idx) // 2]][None])[0]
        im = axh.imshow(win.T, aspect="auto", cmap="Greys", vmin=-2.5, vmax=2.5, origin="lower")
        axh.set_xlabel("time step", labelpad=1); axh.set_ylabel("channel", labelpad=1)
        axh.set_yticks([0, 4, 9]); axh.set_yticklabels(["1", "5", "10"])
        axh.set_xticks([0, 9, 19]); axh.set_xticklabels(["1", "10", "20"])
    save(fig, "fig_signals.png")


def fig_search():
    S = R["search"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    n = np.arange(1, 21)
    for opt, ls, mk in (("BO", "-", "o"), ("RS", "--", "s")):
        ax.plot(n, S[opt]["best_so_far"], color="black", ls=ls, lw=1.0, label=f"{opt} best so far")
        ax.scatter(n, S[opt]["trace"], marker=mk, s=12, facecolor="white" if opt == "RS" else "black", edgecolor="black",
                   lw=0.6, zorder=3, label=f"{opt} candidates")
    ref = S.get("ref", {})
    for k, lab, ls in (("svm", "SVM", ":"), ("attn", "manual", "-.")):
        if k in ref:
            ax.axhline(ref[k], color="0.45", lw=0.8, ls=ls)
            ax.text(0.6, ref[k] + 0.002, lab, va="bottom", ha="left", fontsize=6.3, color="0.3")
    ax.axvline(6.5, color="0.7", lw=0.6)
    ax.text(3.5, 0.03, "shared initial\ndesign", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.2)
    ax.set_xlabel("evaluation"); ax.set_ylabel("validation BA (panel)")
    lo = min(min(S["BO"]["trace"]), min(S["RS"]["trace"]))
    ax.set_ylim(lo - 0.02, max(S["BO"]["best_fitness"], S["RS"]["best_fitness"], *ref.values()) + 0.015)
    ax.set_xlim(0.3, 20.5)
    ax.legend(fontsize=6.3, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), handlelength=1.8, columnspacing=0.8)
    ax = axs[1]
    for opt, mk, fc in (("BO", "o", "black"), ("RS", "s", "white")):
        h = [c["hidden"] * c["layers"] for c in S[opt]["cfgs"]]
        ax.scatter(h, S[opt]["trace"], marker=mk, s=16, facecolor=fc, edgecolor="black", lw=0.6, label=opt)
    ax.set_xlabel(r"units per direction $\times$ layers"); ax.set_ylabel("validation BA (panel)")
    ax.legend(fontsize=6.3, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    save(fig, "fig_search.png")


def fig_main():
    M = R["models"]; ms = [m for m in ORDER if m in M]
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    data = [np.array(M[m]["acc_subject"]) * 100 for m in ms]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.0),
                    whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7), boxprops=dict(lw=0.7))
    for i, (b, m) in enumerate(zip(bp["boxes"], ms)):
        b.set_facecolor("0.85" if m.startswith("ATTN") else "white")
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(i + 1 + rng.uniform(-0.13, 0.13, len(d)), d, s=5, color="black", alpha=0.7, lw=0, zorder=3)
        ax.text(i + 1, ax.get_ylim()[1] if False else 0, "", fontsize=1)
    top = max(max(d) for d in data) + 2
    for i, m in enumerate(ms):
        ax.text(i + 1, top, f"{M[m]['acc'][0] * 100:.1f}", ha="center", va="bottom", fontsize=6.6)
    ax.set_ylim(min(min(d) for d in data) - 3, top + 5)
    ax.set_xticks(range(1, len(ms) + 1)); ax.set_xticklabels([NAME[m] for m in ms], fontsize=7)
    ax.set_ylabel("test accuracy (%)")
    ax.yaxis.grid(True, color="0.88", lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_main.png")


def fig_leak():
    L = R["leak"]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.2), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, (k, lab) in zip(axs, (("ATTN_DEF", "Att-BiLSTM (manual)"), ("SVM", "SVM, penalty 10"))):
        a, b = np.array(L[k]["rep_subject"]) * 100, np.array(L[k]["random_subject"]) * 100
        for u, v in zip(a, b):
            ax.plot([0, 1], [u, v], color="0.55", lw=0.5)
        ax.scatter(np.zeros_like(a), a, s=8, color="black", zorder=3); ax.scatter(np.ones_like(b), b, s=8, color="black", zorder=3)
        ax.plot([0, 1], [a.mean(), b.mean()], color="black", lw=1.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["repetition\nsplit", "random\nwindows"], fontsize=6.8)
        ax.set_xlim(-0.3, 1.3); ax.set_title(f"{lab}\n" + r"$\Delta$" + f" = +{(b - a).mean():.1f} pts", fontsize=7, pad=3)
    axs[0].set_ylabel("test accuracy (%)")
    save(fig, "fig_leak.png")


def fig_faith():
    F = R["faith"]["ATTN_BO"]; K = [1, 2, 4, 6, 8, 10]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), gridspec_kw=dict(wspace=0.32))
    sty = {"attention": ("-", "o", "black"), "saliency": ("--", "s", "black"), "random": (":", "^", "white"), "reverse": ("-.", "v", "white")}
    for ax, kind, yl in ((axs[0], "del", "accuracy after deleting top-K"), (axs[1], "keep", "accuracy keeping only top-K")):
        for meth, (ls, mk, fc) in sty.items():
            ax.plot(K, np.array(F[f"{kind}_{meth}"]) * 100, ls=ls, marker=mk, ms=3.5, mfc=fc, mec="black", color="black", lw=0.9,
                    label={"attention": "attention, high first", "saliency": "gradient × input", "random": "random",
                           "reverse": "attention, low first"}[meth])
        ax.axhline(F["acc"][0] * 100, color="0.6", lw=0.6)
        ax.set_xlabel("K (of 20 time steps)"); ax.set_ylabel(yl + " (%)"); ax.set_xticks(K)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.6, ncol=4, loc="lower center", bbox_to_anchor=(0.36, 0.93), handlelength=2.4, columnspacing=1.2)
    ax = axs[2]; t = np.arange(1, 21)
    ax.plot(t, F["att_profile"], color="black", lw=1.0, marker="o", ms=2.5, label="attention " + r"$\bar{\alpha}_t$")
    ax.plot(t, F["sal_profile"], color="black", lw=1.0, ls="--", marker="s", ms=2.5, mfc="white", label="saliency (normalised)")
    ax.axhline(1 / 20, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel("time step in window"); ax.set_ylabel("mean weight"); ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(fontsize=6.3, loc="upper center")
    ax.set_ylim(0, max(max(F["att_profile"]), max(F["sal_profile"])) * 1.45)
    save(fig, "fig_faith.png")


def fig_loso():
    L = R["loso"]
    keys = [k for k in ("LDA|calib", "SVM|calib", "TCN_DEF|calib", "ATTN_DEF|global", "ATTN_DEF|calib", "ATTN_BO|calib") if k in L]
    lab = {"LDA|calib": "LDA", "SVM|calib": "SVM", "TCN_DEF|calib": "TCN", "ATTN_DEF|global": "Att.\nmanual\nglobal",
           "ATTN_DEF|calib": "Att.\nmanual\ncalib.", "ATTN_BO|calib": "Att.\nBO\ncalib."}
    fig, ax = plt.subplots(figsize=(3.45, 2.4))
    data = [np.array(L[k]["ba_subject"]) * 100 for k in keys]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1.0),
                    whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7), boxprops=dict(lw=0.7))
    for b, k in zip(bp["boxes"], keys):
        b.set_facecolor("0.85" if k.startswith("ATTN") else "white")
    rng = np.random.default_rng(1)
    for i, d in enumerate(data):
        ax.scatter(i + 1 + rng.uniform(-0.12, 0.12, len(d)), d, s=4, color="black", lw=0, alpha=0.7, zorder=3)
    top = max(max(d) for d in data) + 1.5
    for i, k in enumerate(keys):
        ax.text(i + 1, top, f"{L[k]['ba'][0] * 100:.1f}", ha="center", va="bottom", fontsize=6.3)
    ax.set_ylim(0, top + 6)
    ax.set_xticks(range(1, len(keys) + 1)); ax.set_xticklabels([lab[k] for k in keys], fontsize=6.4)
    ax.set_ylabel("LOSO balanced accuracy (%)"); ax.yaxis.grid(True, color="0.88", lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_loso.png")


def fig_robust():
    RB = R["robust"]
    conds = ["clean", "gain_0.5", "gain_2.0", "noise_0.5", "noise_1.0", "drop", "shift_+1", "shift_-1"]
    clab = ["clean", "gain\n×0.5", "gain\n×2", "noise\n0.5", "noise\n1.0", "one\nelectrode\nlost", "rotate\n+1", "rotate\n−1"]
    models = ["SVM", "TCN_DEF", "BILSTM_DEF", "ATTN_DEF", "ATTN_BO"]
    hatch = ["", "////", "....", "xxx", ""]; fc = ["white", "white", "white", "white", "0.35"]
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    wdt = 0.16
    for j, (m, h, f) in enumerate(zip(models, hatch, fc)):
        v = [RB[f"{m}|{c}"][0] * 100 for c in conds]
        ax.bar(np.arange(len(conds)) + (j - 2) * wdt, v, wdt, color=f, edgecolor="black", lw=0.5, hatch=h,
               label=NAME[m].replace("\n", " "))
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(clab, fontsize=6.6)
    ax.set_ylabel("test accuracy (%)"); ax.set_ylim(0, 100)
    ax.legend(fontsize=6.3, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), handlelength=1.6, columnspacing=1.0)
    ax.yaxis.grid(True, color="0.88", lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_robust.png")


def fig_classes():
    C = R["classes"]; rec = np.array(C["recall"]) * 100
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.22))
    ax = axs[0]
    cols = ["0.3"] + ["white"] * 12 + ["0.75"] * 17 + ["0.5"] * 23
    bars = ax.bar(np.arange(53), rec, color=cols, edgecolor="black", lw=0.35, width=0.8)
    bars[0].set_hatch("/////"); bars[0].set_facecolor("white")
    for x0, x1, lab in ((1, 12, "exercise A"), (13, 29, "exercise B"), (30, 52, "exercise C")):
        ax.text((x0 + x1) / 2, 103, lab, ha="center", va="bottom", fontsize=6.6)
        ax.plot([x0 - 0.4, x1 + 0.4], [101.5, 101.5], color="black", lw=0.6)
    ax.text(0, 103, "rest", ha="center", va="bottom", fontsize=6.6)
    ax.set_xlim(-2.2, 53.5); ax.set_ylim(0, 112); ax.set_xticks([0, 12, 29, 52]); ax.set_xticklabels(["0", "12", "29", "52"])
    ax.set_xlabel("class index"); ax.set_ylabel("recall (%)")
    ax = axs[1]
    cm = np.array(C["cm"])
    im = ax.imshow(np.sqrt(cm), cmap="Greys", vmin=0, vmax=1)
    for b in (0.5, 12.5, 29.5):
        ax.axhline(b, color="0.5", lw=0.4); ax.axvline(b, color="0.5", lw=0.4)
    ax.set_xticks([0, 12, 29, 52]); ax.set_yticks([0, 12, 29, 52])
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0", "0.25", "1"]); cb.ax.tick_params(labelsize=6)
    cb.set_label("row-normalised rate", fontsize=6.5)
    save(fig, "fig_classes.png")


def fig_rel():
    Z, Y, ZE, YE = [], [], [], []
    for s in range(1, 28):
        d = np.load(os.path.join(RUN, "final", f"ATTN_BO_S{s}_s0.npz"))
        Z.append(d["z"].astype(float)); Y.append(d["y"])
    Z, Y = np.concatenate(Z), np.concatenate(Y)
    P = softmax(Z); conf = P.max(1); acc = (P.argmax(1) == Y)
    edges = np.linspace(0, 1, 16); cx, ay, cnt = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.sum() > 50:
            cx.append(conf[s].mean()); ay.append(acc[s].mean()); cnt.append(s.mean())
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.7, ls="--", label="perfect calibration")
    ax.bar(cx, ay, width=0.055, color="0.85", edgecolor="black", lw=0.5, label="observed accuracy")
    ax.plot(cx, cx, ls="none", marker="_", color="black", ms=7, label="mean confidence")
    ax2 = ax.twinx(); ax2.plot(cx, np.array(cnt) * 100, color="black", lw=0.9, marker="o", ms=2.5, label="share of windows")
    ax2.set_ylabel("share of windows (%)"); ax2.set_ylim(0, max(cnt) * 100 * 2.2)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.2, loc="upper left")
    save(fig, "fig_rel.png")


def fig_eff():
    M, E = R["models"], R["efficiency"]
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    mk = {"LDA": "v", "SVM": "s", "RF": "D", "TCN_DEF": "^", "BILSTM_DEF": "o", "ATTN_DEF": "o", "ATTN_RS": "o", "ATTN_BO": "o"}
    fcs = {"ATTN_BO": "black", "ATTN_RS": "0.5", "ATTN_DEF": "0.8"}
    off = {"LDA": (6, -2), "SVM": (6, 2), "RF": (-6, -10), "TCN_DEF": (0, 7), "BILSTM_DEF": (0, -12), "ATTN_DEF": (6, -12),
           "ATTN_RS": (7, 2), "ATTN_BO": (7, -4)}
    for m in ORDER:
        if m in M and m in E:
            ax.scatter(E[m]["latency_ms"], M[m]["acc"][0] * 100, marker=mk[m], s=26, facecolor=fcs.get(m, "white"),
                       edgecolor="black", lw=0.7, zorder=3)
            ax.annotate(NAME[m].replace("\n", " "), (E[m]["latency_ms"], M[m]["acc"][0] * 100), textcoords="offset points",
                        xytext=off[m], fontsize=6.0, ha="left" if off[m][0] > 0 else ("right" if off[m][0] < 0 else "center"))
    ax.set_ylim(51, 72.5)
    ax.set_xscale("log"); ax.set_xlabel("single-window CPU latency (ms, log scale)"); ax.set_ylabel("test accuracy (%)")
    ax.yaxis.grid(True, color="0.9", lw=0.5); ax.set_axisbelow(True)
    xl = ax.get_xlim(); ax.set_xlim(xl[0] / 1.5, xl[1] * 3)
    save(fig, "fig_eff.png")


def fig_sens():
    S = R["search"]
    ev = S["BO"]["cfgs"] + S["RS"]["cfgs"][6:]
    fit = S["BO"]["trace"] + S["RS"]["trace"][6:]
    src = ["BO"] * 20 + ["RS"] * 14
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.9), sharey=True, gridspec_kw=dict(wspace=0.12))
    items = [("hidden", r"units per direction $h$", False), ("lr", r"learning rate $\eta$", True),
             ("aug", r"augmentation $\sigma_g$", False), ("att", r"attention width $d_a$", False)]
    for ax, (k, lab, lg) in zip(axs, items):
        for c, f, o in zip(ev, fit, src):
            ax.scatter(c[k], f, s=13, marker="o" if o == "BO" else "s", facecolor="black" if o == "BO" else "white",
                       edgecolor="black", lw=0.6)
        if lg:
            ax.set_xscale("log")
        rho = R["search"]["sens"][k]["rho"]
        ax.set_xlabel(lab + "\n" + r"$r_s$" + f" = {rho:+.2f}")
    axs[0].set_ylabel("validation BA (panel)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="none", mfc="black", mec="black", ms=4, label="BO"),
         Line2D([], [], marker="s", ls="none", mfc="white", mec="black", ms=4, label="RS")]
    fig.legend(handles=h, fontsize=6.6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.9))
    save(fig, "fig_sens.png")


def fig_subject():
    M, L = R["models"], R["loso"]
    b = np.array(M["ATTN_BO"]["acc_subject"]) * 100; s_ = np.array(M["SVM"]["acc_subject"]) * 100
    lo = np.array(L["ATTN_DEF|calib"]["ba_subject"]) * 100
    d = np.array([json.load(open(os.path.join(RUN, "final", f"ATTN_DEF_S{s}_s0.json")))["ba"] for s in range(1, 28)]) * 100
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.9), gridspec_kw=dict(wspace=0.45))
    ax = axs[0]
    ax.scatter(s_, b, s=10, color="black")
    lim = [min(b.min(), s_.min()) - 2, max(b.max(), s_.max()) + 2]
    ax.plot(lim, lim, color="0.6", lw=0.7, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SVM accuracy (%)"); ax.set_ylabel("Att-BiLSTM (BO) (%)")
    ax.text(0.04, 0.96, f"r = {np.corrcoef(s_, b)[0, 1]:.2f}", transform=ax.transAxes, va="top", fontsize=6.6)
    ax = axs[1]
    ax.scatter(d, lo, s=10, color="black")
    ax.set_xlabel("within-subject BA (%)"); ax.set_ylabel("LOSO BA, calibrated (%)")
    ax.text(0.04, 0.96, f"r = {np.corrcoef(d, lo)[0, 1]:.2f}", transform=ax.transAxes, va="top", fontsize=6.6)
    ax.set_ylim(0, max(lo) + 8)
    save(fig, "fig_subject.png")


if __name__ == "__main__":
    which = sys.argv[5].split(",") if len(sys.argv) > 5 else None
    for name, fn in list(globals().items()):
        if name.startswith("fig_") and callable(fn) and (which is None or name in which):
            try:
                fn(); print("ok", name)
            except KeyError as e:
                print("skip", name, "missing", e)
