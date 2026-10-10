"""Collect every stored run into results.json (all numbers in the paper come from here).

usage: python analyze.py RUNDIR DATADIR OUT.json
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
from core import softmax, ece, N_CLASSES, balanced_accuracy, macro_f1

RUN, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SUBJ = list(range(1, 28))
NETS = ["TCN_DEF", "BILSTM_DEF", "ATTN_DEF", "ATTN_RS", "ATTN_BO"]
CLASSIC = ["LDA", "SVM", "RF"]
MODELS = CLASSIC + NETS + ["ATTN_BO_FULL"]
METRICS = ["acc", "acc_mov", "ba", "f1", "ece", "nll"]


def jl(p):
    return json.load(open(p))


def per_subject(model, seed=0):
    out = {}
    for s in SUBJ:
        if model in CLASSIC:
            p = os.path.join(RUN, "final", f"{model}_S{s}.json")
            if os.path.exists(p):
                out[s] = jl(p)["runs"][0]
        else:
            p = os.path.join(RUN, "final", f"{model}_S{s}_s{seed}.json")
            if os.path.exists(p):
                out[s] = jl(p)
    return out


def ms(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0]


def boot_ci(v, B=5000, seed=0):
    v = np.asarray(v, float); rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def rank_biserial(d):
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float(4 / (len(d) * (len(d) + 1)) * r[d > 0].sum() - 1)


def bayes_sr(d, rope=0.01, draws=20000, seed=0):
    """Bayesian signed-rank test (Benavoli et al.) with pseudo-observation 0."""
    d = np.concatenate([[0.0], np.asarray(d, float)])
    n = len(d); rng = np.random.default_rng(seed)
    S = d[:, None] + d[None, :]
    iu = np.triu_indices(n)
    s = S[iu]
    right, left = (s > 2 * rope), (s < -2 * rope)
    W = rng.dirichlet(np.ones(n), draws)
    WW = W[:, iu[0]] * W[:, iu[1]]
    pr, pl = WW[:, right].sum(1), WW[:, left].sum(1)
    pe = WW.sum(1) - pr - pl
    win = np.argmax(np.stack([pl, pe, pr], 1), 1)
    return dict(worse=float(np.mean(win == 0)), rope=float(np.mean(win == 1)), better=float(np.mean(win == 2)))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj.tolist()


R = {}
# ------------------------------------------------------------------ data
info = jl(os.path.join(DATA, "info.json"))
w = [info[str(s)]["windows"] for s in SUBJ]; rest = [info[str(s)]["rest"] for s in SUBJ]
R["data"] = dict(min_windows=int(min(w)), max_windows=int(max(w)), total_windows=int(sum(w)),
                 rest_share=float(sum(rest) / sum(w)))

# ------------------------------------------------------------------ search
S = {}
for opt in ("BO", "RS"):
    p = os.path.join(RUN, f"best_{opt}.json")
    if os.path.exists(p):
        b = jl(p)
        tr = [e["fitness"] for e in b["evaluated"]]
        S[opt] = dict(trace=tr, best_so_far=np.maximum.accumulate(tr).tolist(), best=b["best"]["cfg"],
                      best_fitness=b["best"]["fitness"], best_index=int(np.argmax(tr)) + 1,
                      sec=[e["sec"] for e in b["evaluated"]], params=[e["params"] for e in b["evaluated"]],
                      cfgs=[e["cfg"] for e in b["evaluated"]], panel_ba=b["best"]["ba"])
        S["panel"] = b["panel"]
pr = os.path.join(RUN, "panel_ref.json")
if os.path.exists(pr):
    S["ref"] = jl(pr)["mean"]
if "BO" in S and "RS" in S:
    cf = S["BO"]["cfgs"] + S["RS"]["cfgs"][6:]; ft = S["BO"]["trace"] + S["RS"]["trace"][6:]
    S["sens"] = {}
    for k in ("hidden", "layers", "dropout", "lr", "wd", "att", "aug"):
        rr = stats.spearmanr([c[k] for c in cf], ft)
        S["sens"][k] = dict(rho=float(rr[0]), p=float(rr[1]))
    S["n_unique"] = len(cf)
R["search"] = S

# ------------------------------------------------------------------ main comparison
M, PS = {}, {}
for m in MODELS:
    ps = per_subject(m)
    if len(ps) < len(SUBJ):
        continue
    PS[m] = ps
    rec = {k: ms([ps[s][k] for s in SUBJ]) for k in METRICS if k in ps[1]}
    rec["acc_ci"] = boot_ci([ps[s]["acc"] for s in SUBJ])
    rec["acc_nonpanel"] = ms([ps[s]["acc"] for s in SUBJ if s not in S.get("panel", [])])
    rec["acc_subject"] = [ps[s]["acc"] for s in SUBJ]
    if m in NETS:
        rec["params"] = int(ps[1]["params"]); rec["epochs"] = ms([ps[s]["epochs"] for s in SUBJ])
        rec["train_sec"] = ms([ps[s]["train_sec"] for s in SUBJ])
    else:
        rec["train_sec"] = ms([ps[s]["train_sec"] for s in SUBJ])
    M[m] = rec
R["models"] = M

MAIN = [m for m in CLASSIC + NETS if m in PS]
if "ATTN_BO" in PS:
    ref = "ATTN_BO"; others = [m for m in MAIN if m != ref]
    A = np.array([[PS[m][s]["acc"] for s in SUBJ] for m in MAIN])
    fr = stats.friedmanchisquare(*A)
    ranks = stats.rankdata(-A, axis=0).mean(1)
    R["friedman"] = dict(stat=float(fr.statistic), p=float(fr.pvalue), k=len(MAIN), ranks=dict(zip(MAIN, ranks.tolist())))
    W = {}
    ps_ = []
    for m in others:
        d = np.array([PS[ref][s]["acc"] - PS[m][s]["acc"] for s in SUBJ])
        t = stats.wilcoxon(d) if np.any(d != 0) else None
        W[m] = dict(diff=ms(d), p=float(t.pvalue) if t else 1.0, rrb=rank_biserial(d), wins=int((d > 0).sum()),
                    bayes=bayes_sr(d))
        ps_.append(W[m]["p"])
    for m, a in zip(others, holm(ps_)):
        W[m]["p_holm"] = a
    R["wilcoxon_bo"] = W
    # chain increments
    chain = {}
    for a, b in (("BILSTM_DEF", "ATTN_DEF"), ("ATTN_DEF", "ATTN_BO"), ("ATTN_RS", "ATTN_BO"), ("BILSTM_DEF", "ATTN_BO"),
                 ("TCN_DEF", "ATTN_BO"), ("SVM", "ATTN_BO"), ("RF", "ATTN_BO"), ("ATTN_DEF", "ATTN_RS"),
                 ("ATTN_BO", "ATTN_BO_FULL"), ("SVM", "ATTN_BO_FULL"), ("RF", "ATTN_BO_FULL")):
        if a in PS and b in PS:
            d = np.array([PS[b][s]["acc"] - PS[a][s]["acc"] for s in SUBJ])
            chain[f"{a}->{b}"] = dict(diff=ms(d), p=float(stats.wilcoxon(d).pvalue), bayes=bayes_sr(d), wins=int((d > 0).sum()))
    R["chain"] = chain

# ------------------------------------------------------------------ calibration, classes, confusion
cal, cls = {}, {}


def ex_of(c):
    return "R" if c == 0 else ("A" if c <= 12 else ("B" if c <= 29 else "C"))
for m in NETS:
    if m not in PS:
        continue
    eb, ea, Ts = [], [], []    # temperature fitted on the early-stopping repetition 10
    Zall, Yall = [], []
    for s in SUBJ:
        d = np.load(os.path.join(RUN, "final", f"{m}_S{s}_s0.npz"))
        z, y, ze, ye = d["z"].astype(np.float64), d["y"], d["z_es"].astype(np.float64), d["y_es"]
        grid = np.exp(np.linspace(np.log(0.25), np.log(8), 200))
        nll = [-np.mean(np.log(softmax(ze, t)[np.arange(len(ye)), ye] + 1e-12)) for t in grid]
        T = grid[int(np.argmin(nll))]; Ts.append(T)
        eb.append(ece(softmax(z), y)); ea.append(ece(softmax(z, T), y))
        Zall.append(z); Yall.append(y)
    cal[m] = dict(ece_raw=ms(eb), ece_ts=ms(ea), T=ms(Ts))
    if m == "ATTN_BO":
        Z = np.concatenate(Zall); Y = np.concatenate(Yall); P = Z.argmax(1)
        rec = [float(np.mean(P[Y == c] == c)) for c in range(N_CLASSES)]
        cm = np.zeros((N_CLASSES, N_CLASSES)); np.add.at(cm, (Y, P), 1)
        cmn = cm / cm.sum(1, keepdims=True)
        off = cmn.copy(); np.fill_diagonal(off, 0)
        pairs = sorted([(float(off[i, j] + off[j, i]), i, j) for i in range(N_CLASSES) for j in range(i + 1, N_CLASSES)], reverse=True)[:8]
        ex = {"rest": [0], "A": list(range(1, 13)), "B": list(range(13, 30)), "C": list(range(30, 53))}
        cls = dict(recall=rec, exercise={k: float(np.mean(P[np.isin(Y, v)] == Y[np.isin(Y, v)])) for k, v in ex.items()},
                   pairs=[dict(score=a, i=i, j=j) for a, i, j in pairs], cm=cmn.round(4).tolist(),
                   rest_as_move=float(np.mean(P[Y == 0] != 0)), move_as_rest=float(np.mean(P[Y > 0] == 0)),
                   within_ex_err=float(np.mean([ex_of(a) == ex_of(b) for a, b in zip(Y[(P != Y) & (Y > 0) & (P > 0)], P[(P != Y) & (Y > 0) & (P > 0)])])),
                   within_ex_chance=float(sum(((np.array([ex_of(c) for c in range(1, 53)]) == e).sum() - 1) * (np.array([ex_of(c) for c in range(1, 53)]) == e).sum() for e in "ABC") / (52 * 51)))
R["calibration"] = cal
R["classes"] = cls

# ------------------------------------------------------------------ seeds
seeds = {}
for m in ("ATTN_BO", "ATTN_DEF"):
    subs = [s for s in SUBJ if all(os.path.exists(os.path.join(RUN, "final", f"{m}_S{s}_s{k}.json")) for k in (0, 1, 2))]
    if subs:
        A = np.array([[jl(os.path.join(RUN, "final", f"{m}_S{s}_s{k}.json"))["acc"] for k in (0, 1, 2)] for s in subs])
        seeds[m] = dict(subjects=subs, within_sd=float(A.std(1, ddof=1).mean()), between_sd=float(A.mean(1).std(ddof=1)),
                        seed0=float(A[:, 0].mean()), mean3=float(A.mean()))
if len(seeds) == 2:
    sub = sorted(set(seeds["ATTN_BO"]["subjects"]) & set(seeds["ATTN_DEF"]["subjects"]))
    d = [np.mean([jl(os.path.join(RUN, "final", f"ATTN_BO_S{s}_s{k}.json"))["acc"] -
                  jl(os.path.join(RUN, "final", f"ATTN_DEF_S{s}_s{k}.json"))["acc"] for k in (0, 1, 2)]) for s in sub]
    seeds["diff_mean3"] = ms(d); seeds["diff_wins"] = int(np.sum(np.array(d) > 0))
R["seeds"] = seeds

# ------------------------------------------------------------------ leakage
L = {}
for m, ref in (("ATTN_DEF", "ATTN_DEF"), ("SVM", "SVMREP")):
    fs = [os.path.join(RUN, "leak", f"{m}_S{s}.json") for s in SUBJ]
    fr_ = [os.path.join(RUN, "leak", f"{ref}_S{s}.json") for s in SUBJ]
    if all(os.path.exists(f) for f in fs) and (ref in PS or all(os.path.exists(f) for f in fr_)):
        a = np.array([jl(f)["acc"] for f in fs])
        b = np.array([PS[ref][s]["acc"] for s in SUBJ]) if ref in PS else np.array([jl(f)["acc"] for f in fr_])
        L[m] = dict(random=ms(a), rep=ms(b), delta=ms(a - b), p=float(stats.wilcoxon(a - b).pvalue),
                    random_subject=a.tolist(), rep_subject=b.tolist())
ov = [os.path.join(RUN, "leak", f"OVERLAP_S{s}.json") for s in SUBJ]
if all(os.path.exists(f) for f in ov) and "SVM" in L:
    O = [jl(f) for f in ov]
    L["overlap"] = {"10": dict(random=L["SVM"]["random"], rep=L["SVM"]["rep"], delta=L["SVM"]["delta"])}
    for st in ("5", "2"):
        a = np.array([o[f"random_{st}"] for o in O]); b = np.array([o[f"rep_{st}"] for o in O])
        L["overlap"][st] = dict(random=ms(a), rep=ms(b), delta=ms(a - b))
R["leak"] = L

# ------------------------------------------------------------------ LOSO
LO = {}
for f in glob.glob(os.path.join(RUN, "loso", "*_S*.json")):
    d = jl(f); LO.setdefault(f"{d['model']}|{d['norm']}", {})[d["subject"]] = d
LS = {}
for k, v in LO.items():
    if len(v) == len(SUBJ):
        LS[k] = dict(acc=ms([v[s]["acc"] for s in SUBJ]), ba=ms([v[s]["ba"] for s in SUBJ]), f1=ms([v[s]["f1"] for s in SUBJ]),
                     acc_subject=[v[s]["acc"] for s in SUBJ], acc_ci=boot_ci([v[s]["acc"] for s in SUBJ]),
                     ba_subject=[v[s]["ba"] for s in SUBJ], ba_ci=boot_ci([v[s]["ba"] for s in SUBJ]),
                     epochs=[v[s].get("epochs") for s in SUBJ])
if "ATTN_BO|calib" in LS:
    for k in LS:
        if k != "ATTN_BO|calib":
            d = np.array(LS["ATTN_BO|calib"]["acc_subject"]) - np.array(LS[k]["acc_subject"])
            db = np.array(LS["ATTN_BO|calib"]["ba_subject"]) - np.array(LS[k]["ba_subject"])
            LS[k]["vs_bo_calib"] = dict(diff=ms(d), p=float(stats.wilcoxon(d).pvalue), wins=int((d > 0).sum()),
                                        diff_ba=ms(db), p_ba=float(stats.wilcoxon(db).pvalue), wins_ba=int((db > 0).sum()))
if "ATTN_DEF|calib" in LS and "ATTN_DEF|global" in LS:
    for key in ("acc", "ba"):
        d = np.array(LS["ATTN_DEF|calib"][f"{key}_subject"]) - np.array(LS["ATTN_DEF|global"][f"{key}_subject"])
        LS["ATTN_DEF|calib"][f"vs_global_{key}"] = dict(diff=ms(d), p=float(stats.wilcoxon(d).pvalue), wins=int((d > 0).sum()))
# majority-class (rest) baseline of the LOSO test sets
rs_c, rs_g = [], []
for s in SUBJ:
    d = np.load(os.path.join(DATA, f"S{s}.npz")); y, r = d["y"], d["rep"]
    rs_c.append(float(np.mean(y[r != 1] == 0))); rs_g.append(float(np.mean(y == 0)))
R["loso_rest"] = dict(calib=ms(rs_c), glob=ms(rs_g), calib_subject=rs_c)
if LS:
    R["loso_below_rest"] = {k: int(np.sum(np.array(v["acc_subject"]) <= np.array(rs_c if k.endswith("calib") else rs_g))) for k, v in LS.items()}
# training-set size per movement class (fit reps of the final protocol)
cnt = []
for s in SUBJ:
    d = np.load(os.path.join(DATA, f"S{s}.npz")); y, r = d["y"], d["rep"]
    f = np.isin(r, (1, 3, 4, 6, 8, 9))
    cnt.append(np.bincount(y[f], minlength=53))
cnt = np.array(cnt)
R["fit_counts"] = dict(per_movement=float(cnt[:, 1:].mean()), rest=float(cnt[:, 0].mean()))
R["loso"] = LS
if "ATTN_DEF|calib" in LS and "ATTN_DEF" in PS and "SVM" in PS and "ATTN_BO" in PS:
    a = np.array([PS["ATTN_BO"][s]["acc"] for s in SUBJ]); v = np.array([PS["SVM"][s]["acc"] for s in SUBJ])
    w_ = np.array([PS["ATTN_DEF"][s]["ba"] for s in SUBJ]); l_ = np.array(LS["ATTN_DEF|calib"]["ba_subject"])
    R["subject_corr"] = dict(svm_bo=dict(zip(("r", "p"), map(float, stats.pearsonr(v, a)))),
                             within_loso=dict(zip(("r", "p"), map(float, stats.pearsonr(w_, l_)))),
                             rho_within_loso=float(stats.spearmanr(w_, l_)[0]))

# ------------------------------------------------------------------ faithfulness
fp = os.path.join(RUN, "faith.json")
if os.path.exists(fp):
    F = jl(fp); FA = {}
    for tag, rows in F.items():
        e = {"acc": ms([r["acc"] for r in rows]), "entropy": ms([r["entropy"] for r in rows]),
             "rank_corr": ms([r["rank_corr"] for r in rows]),
             "att_profile": np.mean([r["att_profile"] for r in rows], 0).tolist(),
             "sal_profile": np.mean([r["sal_profile"] for r in rows], 0).tolist()}
        for kind in ("del", "keep"):
            for meth in ("attention", "saliency", "random", "reverse"):
                e[f"{kind}_{meth}"] = [ms([r[f"{kind}_{meth}_{k}_acc"] for r in rows])[0] for k in (1, 2, 4, 6, 8, 10)]
                if kind == "del":
                    e[f"dp_{meth}"] = [ms([r[f"del_{meth}_{k}_dp"] for r in rows])[0] for k in (1, 2, 4, 6, 8, 10)]
        for k in (2, 4, 6):
            a = np.array([r[f"del_attention_{k}_acc"] for r in rows]); b = np.array([r[f"del_random_{k}_acc"] for r in rows])
            c = np.array([r[f"del_saliency_{k}_acc"] for r in rows])
            e[f"test_att_vs_rand_{k}"] = dict(diff=ms(b - a), p=float(stats.wilcoxon(b - a).pvalue), wins=int(((b - a) > 0).sum()))
            e[f"test_sal_vs_att_{k}"] = dict(diff=ms(a - c), p=float(stats.wilcoxon(a - c).pvalue), wins=int(((a - c) > 0).sum()))
        FA[tag] = e
    R["faith"] = FA

# ------------------------------------------------------------------ robustness
rp = os.path.join(RUN, "robust.json")
if os.path.exists(rp):
    RB = jl(rp); keys = sorted({k for v in RB.values() for k in v})
    RR = {}
    for k in keys:
        RR[k] = ms([RB[str(s)][k] for s in SUBJ])
    R["robust"] = RR

# ------------------------------------------------------------------ efficiency
ep = os.path.join(RUN, "efficiency.json")
if os.path.exists(ep):
    R["efficiency"] = jl(ep)

json.dump(R, open(OUT, "w"), indent=1)
print("models:", {m: round(v["acc"][0], 4) for m, v in M.items()})
if "friedman" in R:
    print("friedman", R["friedman"]["p"], "wilcoxon:", {m: (round(v["diff"][0], 4), round(v["p_holm"], 4)) for m, v in R["wilcoxon_bo"].items()})
print("leak", {k: v.get("delta") for k, v in L.items()})
print("loso", {k: v["acc"] for k, v in LS.items()})
print("search", {k: (v["best_fitness"], v["best_index"]) for k, v in S.items() if isinstance(v, dict) and "best_fitness" in v})
