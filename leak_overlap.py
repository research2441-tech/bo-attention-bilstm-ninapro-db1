"""Leakage versus window overlap (SVM, C = 10).

Windows are re-cut from the raw DB1 files with steps of 5 and 2 samples (75 % and 90 %
overlap) and randomly subsampled to the number of windows obtained with the standard step
of 10, so that only the overlap between neighbouring windows changes.  Each set is
evaluated with the random window partition and with the repetition split.

usage: python leak_overlap.py RAW_DIR RUNDIR
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import prep_db1
from core import features, mask, balanced_accuracy, load_subject, TEST_REPS
from leak import split


def assign_rest(r):
    if (r == 0).any():
        i0 = np.where(r == 0)[0]; nz = np.where(r > 0)[0]
        r[i0] = r[nz[np.clip(np.searchsorted(nz, i0), 0, len(nz) - 1)]]
    return r


def svm_acc(F, y, tr, te):
    ss = StandardScaler().fit(F[tr])
    p = SVC(C=10, gamma="scale", class_weight="balanced").fit(ss.transform(F[tr]), y[tr]).predict(ss.transform(F[te]))
    return float(np.mean(p == y[te]))


def main(raw, rundir):
    od = os.path.join(rundir, "leak"); os.makedirs(od, exist_ok=True)
    for s in range(1, 28):
        out = os.path.join(od, f"OVERLAP_S{s}.json")
        if os.path.exists(out):
            continue
        n_ref = len(load_subject(s)[1]); rec = dict(subject=s, n=n_ref)
        for step in (5, 2):
            prep_db1.STEP = step
            X, y, r = prep_db1.subject(raw, s)
            r = assign_rest(r)
            keep = np.sort(np.random.default_rng(s).choice(len(y), n_ref, replace=False))
            X, y, r = X[keep], y[keep], r[keep]
            F = features(X)
            fit, es, te = split(y, seed=s)
            rec[f"random_{step}"] = svm_acc(F, y, np.concatenate([fit, es]), te)
            rec[f"rep_{step}"] = svm_acc(F, y, np.where(~mask(r, TEST_REPS))[0], np.where(mask(r, TEST_REPS))[0])
        json.dump(rec, open(out, "w"), indent=1)
        print(rec, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
