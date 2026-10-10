"""Group-level hyperparameter search for the attention Bi-LSTM.

A panel of tuning subjects is drawn once with a fixed seed.  Each candidate is trained on
repetitions 1, 3, 4, 6 and 8 of every panel subject and scored by the mean balanced
accuracy on repetitions 9 and 10; the test repetitions 2, 5 and 7 are never touched.
Random search and Bayesian optimisation receive the same budget and share the same six
initial candidates (common random numbers), so the comparison is paired.  The initial
design is evaluated once (mode INIT, split over workers) and cached for both optimisers.

usage: python tune.py INIT WORKER NWORKERS OUTDIR
       python tune.py {BO,RS} OUTDIR
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from core import *

BUDGET, N_INIT, PANEL_SIZE = 20, 6, 3
PANEL = sorted(int(s) for s in np.random.default_rng(2026).choice(np.arange(1, 28), PANEL_SIZE, replace=False))
INNER_TR, INNER_VA = (1, 3, 4, 6, 8), (9, 10)
SEARCH_EPOCHS, SEARCH_PATIENCE = 40, 5
U = np.random.default_rng(17).random((BUDGET, DIM))


def panel_data():
    out = []
    for s in PANEL:
        X, y, r = load_subject(s)
        tr, va = mask(r, INNER_TR), mask(r, INNER_VA)
        Xs = Scaler().fit(X[tr])(X)
        out.append((Xs[tr], y[tr], Xs[va], y[va]))
    return out


def key(u):
    return tuple(np.round(np.asarray(u, float), 8))


def load_cache(outdir):
    cache = {}
    for fn in os.listdir(outdir):
        if fn.startswith("tune_") and fn.endswith(".jsonl"):
            for l in open(os.path.join(outdir, fn)):
                d = json.loads(l); cache[key(d["u"])] = d
    return cache


def make_f(tag, data, outdir, cache, order):
    log = open(os.path.join(outdir, f"tune_{tag}.jsonl"), "a")

    def f(u):
        k = key(u)
        if k not in cache:
            cfg = decode(u); t = time.time(); bas, eps = [], []
            for (Xtr, ytr, Xva, yva) in data:
                m, ep = train("attn", cfg, Xtr, ytr, Xva, yva, seed=0,
                              max_epochs=SEARCH_EPOCHS, patience=SEARCH_PATIENCE)
                bas.append(balanced_accuracy(yva, logits(m, Xva).argmax(1))); eps.append(ep)
            rec = dict(u=[float(v) for v in u], cfg=cfg, fitness=float(np.mean(bas)), ba=bas,
                       epochs=eps, params=n_params(m), sec=time.time() - t)
            log.write(json.dumps(rec) + "\n"); log.flush(); cache[k] = rec
            print(tag, round(rec["fitness"], 4), cfg, round(rec["sec"]), flush=True)
        order.append(cache[k])
        return cache[k]["fitness"]
    return f


def main():
    if sys.argv[1] == "INIT":
        w, nw, outdir = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        os.makedirs(outdir, exist_ok=True)
        f = make_f(f"INIT{w}", panel_data(), outdir, load_cache(outdir), [])
        for i in range(w, N_INIT, nw):
            f(U[i])
        return
    opt, outdir = sys.argv[1], sys.argv[2]
    data, cache, order = panel_data(), load_cache(outdir), []
    f = make_f(opt, data, outdir, cache, order)
    if opt == "RS":
        trace = [f(u) for u in U]
    else:
        trace = BayesOpt(DIM, BUDGET, seed=17, n_startup=N_INIT, init=U[:N_INIT]).run(f)
    best = max(order, key=lambda d: d["fitness"])
    json.dump(dict(panel=PANEL, opt=opt, trace=trace, evaluated=order, best=best),
              open(os.path.join(outdir, f"best_{opt}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
