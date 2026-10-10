"""Parameter counts and single-window CPU latency (one thread) of every model.

usage: python efficiency.py RUNDIR
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from core import *

rundir = sys.argv[1]
torch.set_num_threads(1)
X, y, r = load_subject(1)
tr = mask(r, TRAIN_REPS)
sc = Scaler().fit(X[tr])
x1 = X[mask(r, TEST_REPS)][:1]
REPS = 300


def timeit(fn):
    for _ in range(20):
        fn()
    t = time.perf_counter()
    for _ in range(REPS):
        fn()
    return (time.perf_counter() - t) / REPS * 1e3


out = {}
cfgs = {"BILSTM_DEF": ("bilstm", DEFAULT), "ATTN_DEF": ("attn", DEFAULT), "TCN_DEF": ("tcn", DEFAULT)}
for opt in ("BO", "RS"):
    p = os.path.join(rundir, f"best_{opt}.json")
    if os.path.exists(p):
        cfgs[f"ATTN_{opt}"] = ("attn", json.load(open(p))["best"]["cfg"])
for tag, (kind, cfg) in cfgs.items():
    m = build(kind, cfg, 10).eval()
    with torch.no_grad():
        ms_ = timeit(lambda: m(torch.from_numpy(sc(x1))))
    out[tag] = dict(params=n_params(m), latency_ms=ms_)
F = features(X[tr]); ss = StandardScaler().fit(F)
for name, clf in (("LDA", LinearDiscriminantAnalysis()), ("SVM", SVC(C=10, gamma="scale", class_weight="balanced")),
                  ("RF", RandomForestClassifier(300, min_samples_leaf=2, n_jobs=1, random_state=0))):
    clf.fit(ss.transform(F), y[tr])
    out[name] = dict(latency_ms=timeit(lambda: clf.predict(ss.transform(features(x1)))),
                     n_sv=int(clf.n_support_.sum()) if name == "SVM" else None)
json.dump(out, open(os.path.join(rundir, "efficiency.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
