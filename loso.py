"""Leave-one-subject-out evaluation.

For each held-out subject the model is trained on the other 26 subjects.  From every
training subject N_PER windows are drawn at random (stratified by class) for fitting and
N_ES further windows for early stopping.  Two normalisation schemes are compared:
  calib  - every subject is standardised with the log-domain statistics of its own
           repetition-1 windows (an unlabelled calibration recording); the held-out
           subject is then tested on repetitions 2-10;
  global - one scaler fitted on the pooled training windows is applied to everyone.
The held-out subject is never used for fitting, early stopping or model selection.

usage: python loso.py MODEL NORM RUNDIR [subjects]
       MODEL in {ATTN_BO, ATTN_DEF, BILSTM_DEF, TCN_DEF, LDA, SVM}; NORM in {calib, global}
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from core import *

N_PER, N_ES = 480, 120
KIND = dict(ATTN_BO="attn", ATTN_DEF="attn", BILSTM_DEF="bilstm", TCN_DEF="tcn")


def draw(y, n, rng):
    """Class-stratified random subsample of n indices."""
    idx = []
    cls = np.unique(y)
    per = max(1, n // len(cls))
    for c in cls:
        j = np.where(y == c)[0]
        idx.append(rng.choice(j, min(per, len(j)), replace=False))
    return np.concatenate(idx)


def subject_views(norm):
    """Return, per subject, normalised windows, labels and repetitions."""
    raw = {s: load_subject(s) for s in range(1, 28)}
    views = {}
    for s, (X, y, r) in raw.items():
        if norm == "calib":
            Xs = Scaler().fit(X[r == 1])(X)
        else:
            Xs = X
        views[s] = (Xs, y, r)
    return views


def main(model, norm, rundir, subs):
    od = os.path.join(rundir, "loso"); os.makedirs(od, exist_ok=True)
    views = subject_views(norm)
    cfg = dict(DEFAULT) if not model.endswith("_BO") else json.load(open(os.path.join(rundir, "best_BO.json")))["best"]["cfg"]
    for s in subs:
        out = os.path.join(od, f"{model}_{norm}_S{s}.json")
        if os.path.exists(out):
            continue
        rng = np.random.default_rng(1000 + s)
        Xf, yf, Xe, ye = [], [], [], []
        for t, (X, y, r) in views.items():
            if t == s:
                continue
            i = draw(y, N_PER + N_ES, rng); rng.shuffle(i)
            k = int(len(i) * N_PER / (N_PER + N_ES))
            Xf.append(X[i[:k]]); yf.append(y[i[:k]]); Xe.append(X[i[k:]]); ye.append(y[i[k:]])
        Xf, yf, Xe, ye = map(np.concatenate, (Xf, yf, Xe, ye))
        Xt, yt, rt = views[s]
        te = rt != 1 if norm == "calib" else np.ones(len(yt), bool)
        Xt, yt = Xt[te], yt[te]
        t0 = time.time()
        if model in ("LDA", "SVM"):
            lg = norm == "calib"
            Ff = np.vstack([features(Xf, lg), features(Xe, lg)]); Ft = features(Xt, lg)
            ss = StandardScaler().fit(Ff)
            clf = LinearDiscriminantAnalysis() if model == "LDA" else SVC(C=10, gamma="scale", class_weight="balanced")
            clf.fit(ss.transform(Ff), np.concatenate([yf, ye]))
            p = clf.predict(ss.transform(Ft))
            rec = dict(acc=float(np.mean(p == yt)), ba=balanced_accuracy(yt, p), f1=macro_f1(yt, p))
        else:
            if norm == "global":
                sc = Scaler().fit(Xf); Xf, Xe, Xt = sc(Xf), sc(Xe), sc(Xt)
            m, ep = train(KIND[model], cfg, Xf, yf, Xe, ye, seed=0, max_epochs=40, patience=5)
            rec = dict(epochs=ep, **metrics(logits(m, Xt), yt))
        rec.update(subject=s, model=model, norm=norm, sec=time.time() - t0, n_fit=int(len(yf)))
        json.dump(rec, open(out, "w"), indent=1)
        print(model, norm, s, round(rec["acc"], 4), round(rec["sec"]), flush=True)


if __name__ == "__main__":
    subs = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else range(1, 28)
    main(sys.argv[1], sys.argv[2], sys.argv[3], subs)
