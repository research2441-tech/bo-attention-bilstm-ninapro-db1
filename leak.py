"""Leakage audit: random window-level partition versus the repetition split.

The random partition keeps the proportions of the repetition protocol (6/10 of the windows
for fitting, 1/10 for early stopping, 3/10 for testing) but assigns windows at random,
stratified by class, so that overlapping and temporally adjacent windows of the same
repetition fall on both sides.  Everything else (scaler, model, training) is unchanged.

Mode SVMREP fits the same fixed-penalty SVM on the repetition split (train 1, 3, 4, 6, 8,
9, 10; test 2, 5, 7) so that the SVM comparison differs only in the partition.

usage: python leak.py MODEL RUNDIR      MODEL in {ATTN_DEF, SVM, SVMREP}
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from core import *


def split(y, seed):
    idx = np.arange(len(y))
    rest, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
    fit, es = train_test_split(rest, test_size=1 / 7, stratify=y[rest], random_state=seed)
    return fit, es, te


def main(model, rundir):
    od = os.path.join(rundir, "leak"); os.makedirs(od, exist_ok=True)
    for s in range(1, 28):
        out = os.path.join(od, f"{model}_S{s}.json")
        if os.path.exists(out):
            continue
        X, y, r = load_subject(s)
        fit, es, te = split(y, seed=s)
        if model == "SVMREP":
            fit, es, te = np.where(mask(r, (1, 3, 4, 6, 8, 9)))[0], np.where(mask(r, (10,)))[0], np.where(mask(r, TEST_REPS))[0]
        t = time.time()
        if model in ("SVM", "SVMREP"):
            F = features(X); tr = np.concatenate([fit, es])
            ss = StandardScaler().fit(F[tr])
            clf = SVC(C=10, gamma="scale", class_weight="balanced").fit(ss.transform(F[tr]), y[tr])
            p = clf.predict(ss.transform(F[te]))
            rec = dict(acc=float(np.mean(p == y[te])), ba=balanced_accuracy(y[te], p), f1=macro_f1(y[te], p))
        else:
            Xs = Scaler().fit(X[np.concatenate([fit, es])])(X)
            m, ep = train("attn", DEFAULT, Xs[fit], y[fit], Xs[es], y[es], seed=0)
            rec = dict(epochs=ep, **metrics(logits(m, Xs[te]), y[te]))
        rec.update(subject=s, model=model, sec=time.time() - t)
        json.dump(rec, open(out, "w"), indent=1)
        print(model, s, round(rec["acc"], 4), round(rec["sec"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
