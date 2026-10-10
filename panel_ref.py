"""Reference scores on the tuning panel (validation repetitions 9, 10): the manually chosen
configuration of the attention Bi-LSTM, the plain Bi-LSTM, the TCN and the SVM.

usage: python panel_ref.py RUNDIR
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from core import *
from tune import PANEL, INNER_TR, INNER_VA, SEARCH_EPOCHS, SEARCH_PATIENCE

out = {}
for s in PANEL:
    X, y, r = load_subject(s)
    tr, va = mask(r, INNER_TR), mask(r, INNER_VA)
    Xs = Scaler().fit(X[tr])(X)
    for kind in ("attn", "bilstm", "tcn"):
        m, _ = train(kind, DEFAULT, Xs[tr], y[tr], Xs[va], y[va], seed=0, max_epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE)
        out.setdefault(kind, []).append(balanced_accuracy(y[va], logits(m, Xs[va]).argmax(1)))
    F = features(X); ss = StandardScaler().fit(F[tr])
    p = SVC(C=10, gamma="scale", class_weight="balanced").fit(ss.transform(F[tr]), y[tr]).predict(ss.transform(F[va]))
    out.setdefault("svm", []).append(balanced_accuracy(y[va], p))
    print(s, {k: round(v[-1], 4) for k, v in out.items()}, flush=True)
json.dump(dict(panel=PANEL, scores=out, mean={k: float(np.mean(v)) for k, v in out.items()}),
          open(os.path.join(sys.argv[1], "panel_ref.json"), "w"), indent=1)
