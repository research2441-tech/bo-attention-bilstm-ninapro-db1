"""Hand-crafted feature baselines (LDA, RBF-SVM, random forest) under the repetition split.

Fifty window features (core.features) are standardised on the training repetitions.  The
SVM penalty C is chosen per subject on an inner split (train 1, 3, 4, 6, 8; validate 9, 10);
LDA and the random forest use fixed settings.  The final classifier is fitted on all seven
training repetitions.

usage: python classical.py RUNDIR
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from core import *

C_GRID = (1.0, 10.0, 100.0)


def make(name, C=10.0):
    if name == "LDA":
        return LinearDiscriminantAnalysis()
    if name == "SVM":
        return SVC(C=C, kernel="rbf", gamma="scale", class_weight="balanced", decision_function_shape="ovr")
    return RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=0)


def scores(clf, F):
    if hasattr(clf, "predict_proba") and not isinstance(clf, SVC):
        P = np.full((len(F), N_CLASSES), 1e-6); P[:, clf.classes_] = clf.predict_proba(F)
        return np.log(P)
    Z = np.full((len(F), N_CLASSES), -1e3); Z[:, clf.classes_] = clf.decision_function(F)
    return Z


def main(rundir):
    os.makedirs(os.path.join(rundir, "final"), exist_ok=True)
    for s in range(1, 28):
        X, y, r = load_subject(s)
        F = features(X)
        tr, te = mask(r, TRAIN_REPS), mask(r, TEST_REPS)
        itr, iva = mask(r, (1, 3, 4, 6, 8)), mask(r, (9, 10))
        for name in ("LDA", "SVM", "RF"):
            out = os.path.join(rundir, "final", f"{name}_S{s}.json")
            if os.path.exists(out):
                continue
            t = time.time(); C = None
            if name == "SVM":
                ss = StandardScaler().fit(F[itr]); best = -1
                for c in C_GRID:
                    ba = balanced_accuracy(y[iva], make("SVM", c).fit(ss.transform(F[itr]), y[itr]).predict(ss.transform(F[iva])))
                    if ba > best:
                        best, C = ba, c
            ss = StandardScaler().fit(F[tr])
            clf = make(name, C or 10.0).fit(ss.transform(F[tr]), y[tr])
            z = scores(clf, ss.transform(F[te]))
            rec = dict(seed=0, train_sec=time.time() - t, C=C, **metrics(z, y[te]))
            np.savez_compressed(out.replace(".json", ".npz"), z=z[None].astype(np.float16), y=y[te])
            json.dump(dict(tag=name, subject=s, runs=[rec]), open(out, "w"), indent=1)
            print(name, s, round(rec["acc"], 4), round(rec["train_sec"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
