"""Attention faithfulness and robustness probes on the saved seed-0 models.

Faithfulness (attention models only): for every test window the k time steps ranked
highest by (a) attention, (b) gradient x input saliency, (c) a random order and
(d) lowest attention are replaced by the training mean (zero in the standardised domain).
The drop in accuracy and in the probability of the originally predicted class is recorded
for k = 1, 2, 4, 6, 8, 10 (deletion), and the accuracy when only the top-k steps are kept
is recorded as sufficiency.

Robustness (all saved networks and the SVM): test windows are perturbed in the raw RMS
domain before standardisation - global gain change, additive noise proportional to the
channel median, loss of one electrode (channel set to zero) and a one-position rotation of
the eight equally spaced forearm electrodes (channels 1-8).  The robustness probe uses a
random subset of N_ROBUST test windows per subject.

usage: python probe.py {faith,robust} RUNDIR
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from core import *

FIT_REPS, ES_REPS = (1, 3, 4, 6, 8, 9), (10,)
KS = (1, 2, 4, 6, 8, 10)
N_ROBUST = 2000          # random test windows per subject used for the robustness probe
NETS = dict(ATTN_BO="attn", ATTN_DEF="attn", BILSTM_DEF="bilstm", TCN_DEF="tcn")


def load_model(tag, s, rundir):
    cfg = json.load(open(os.path.join(rundir, "final", f"{tag}_S{s}_s0.json")))["cfg"]
    m = build(NETS[tag], cfg, 10)
    m.load_state_dict(torch.load(os.path.join(rundir, "final", f"{tag}_S{s}_seed0.pt"), weights_only=True))
    return m.eval()


def subject_test(s):
    X, y, r = load_subject(s)
    fit, es, te = mask(r, FIT_REPS), mask(r, ES_REPS), mask(r, TEST_REPS)
    return X, y, Scaler().fit(X[fit | es]), te, fit | es


def grad_x_input(m, Xs, cls):
    x = torch.from_numpy(Xs).requires_grad_(True)
    out = m(x)
    out.gather(1, torch.from_numpy(cls)[:, None]).sum().backward()
    return (x.grad * x).abs().sum(2).detach().numpy()          # (n, T)


def masked(Xs, order, k, keep=False):
    Z = Xs.copy()
    n, T = Xs.shape[:2]
    sel = np.zeros((n, T), bool)
    np.put_along_axis(sel, order[:, :k], True, 1)
    if keep:
        sel = ~sel
    Z[sel] = 0.0
    return Z


def faith(rundir):
    res = {}
    for tag in ("ATTN_BO", "ATTN_DEF"):
        rows = []
        for s in range(1, 28):
            X, y, sc, te, _ = subject_test(s)
            Xs, yt = sc(X[te]), y[te]
            m = load_model(tag, s, rundir)
            z, a = logits(m, Xs, return_att=True)
            pred = z.argmax(1); p0 = softmax(z)[np.arange(len(yt)), pred]
            g = np.concatenate([grad_x_input(m, Xs[i:i + 2048], pred[i:i + 2048]) for i in range(0, len(Xs), 2048)])
            rng = np.random.default_rng(s)
            orders = dict(attention=np.argsort(-a, 1), saliency=np.argsort(-g, 1),
                          random=np.argsort(rng.random(a.shape), 1), reverse=np.argsort(a, 1))
            row = dict(subject=s, acc=float(np.mean(pred == yt)),
                       entropy=float(np.mean(-(a * np.log(a + 1e-12)).sum(1)) / np.log(a.shape[1])),
                       att_profile=a.mean(0).tolist(), sal_profile=(g / g.sum(1, keepdims=True)).mean(0).tolist(),
                       rank_corr=float(np.mean([np.corrcoef(np.argsort(np.argsort(a[i])), np.argsort(np.argsort(g[i])))[0, 1]
                                                for i in range(0, len(a), 7)])))
            for name, o in orders.items():
                for k in KS:
                    zd = logits(m, masked(Xs, o, k))
                    row[f"del_{name}_{k}_acc"] = float(np.mean(zd.argmax(1) == yt))
                    row[f"del_{name}_{k}_dp"] = float(np.mean(p0 - softmax(zd)[np.arange(len(yt)), pred]))
                    zk = logits(m, masked(Xs, o, k, keep=True))
                    row[f"keep_{name}_{k}_acc"] = float(np.mean(zk.argmax(1) == yt))
            rows.append(row)
            print(tag, s, round(row["acc"], 3), round(row["del_attention_4_acc"], 3), round(row["del_random_4_acc"], 3),
                  round(row["del_saliency_4_acc"], 3), flush=True)
        res[tag] = rows
    json.dump(res, open(os.path.join(rundir, "faith.json"), "w"))


PERTURB = [("clean", None)] + [(f"gain_{g}", ("gain", g)) for g in (0.5, 0.8, 1.25, 2.0)] + \
          [(f"noise_{v}", ("noise", v)) for v in (0.25, 0.5, 1.0)] + \
          [("drop", ("drop", None)), ("shift_+1", ("shift", 1)), ("shift_-1", ("shift", -1))]


def perturb(X, kind, v, med, rng):
    if kind == "gain":
        return X * v
    if kind == "noise":
        return np.abs(X + rng.normal(size=X.shape).astype(np.float32) * v * med[None, None, :])
    if kind == "shift":
        Z = X.copy(); Z[:, :, :8] = np.roll(X[:, :, :8], v, axis=2); return Z
    raise ValueError(kind)


def robust(rundir):
    out = {}
    for s in range(1, 28):
        X, y, sc, te, trm = subject_test(s)
        idx = np.sort(np.random.default_rng(100 + s).choice(np.where(te)[0], N_ROBUST, replace=False))
        Xt, yt = X[idx], y[idx]
        med = np.median(X[trm].reshape(-1, 10), 0)
        F = features(X[trm]); ss = StandardScaler().fit(F)
        svm = SVC(C=10, gamma="scale", class_weight="balanced").fit(ss.transform(F), y[trm])
        models = {tag: load_model(tag, s, rundir) for tag in NETS}
        rng = np.random.default_rng(s)
        rec = {}
        for name, spec in PERTURB:
            if spec is None:
                variants = [Xt]
            elif spec[0] == "drop":
                variants = []
                for c in range(10):
                    Z = Xt.copy(); Z[:, :, c] = 0.0; variants.append(Z)
            else:
                variants = [perturb(Xt, spec[0], spec[1], med, rng)]
            for tag, m in models.items():
                rec[f"{tag}|{name}"] = float(np.mean([np.mean(logits(m, sc(V)).argmax(1) == yt) for V in variants]))
            rec[f"SVM|{name}"] = float(np.mean([np.mean(svm.predict(ss.transform(features(V))) == yt) for V in variants]))
        out[s] = rec
        print(s, {k: round(v, 3) for k, v in rec.items() if k.endswith("clean") or k.endswith("shift_+1")}, flush=True)
    json.dump(out, open(os.path.join(rundir, "robust.json"), "w"), indent=1)


if __name__ == "__main__":
    {"faith": faith, "robust": robust}[sys.argv[1]](sys.argv[2])
