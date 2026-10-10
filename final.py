"""Per-subject training and testing under the repetition split.

Repetitions 1, 3, 4, 6, 8 and 9 train the network, repetition 10 drives early stopping and
repetitions 2, 5 and 7 are the test set.  The channel scaler is fitted on the seven
training repetitions.  One file is written per (model, subject, seed).

Tags: BILSTM_DEF, ATTN_DEF, TCN_DEF use the default configuration; ATTN_BO and ATTN_RS use
the configuration selected by tune.py.  ATTN_BO_FULL refits the ATTN_BO configuration on all
seven training repetitions for the number of epochs selected by early stopping in ATTN_BO.

usage: python final.py TAG RUNDIR [subjects] [seeds]
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from core import *

FIT_REPS, ES_REPS = (1, 3, 4, 6, 8, 9), (10,)
KIND = dict(BILSTM_DEF="bilstm", ATTN_DEF="attn", TCN_DEF="tcn", ATTN_BO="attn", ATTN_RS="attn", ATTN_BO_FULL="attn")


def config(tag, rundir):
    if tag.endswith("_DEF"):
        return dict(DEFAULT)
    return json.load(open(os.path.join(rundir, f"best_{tag.split('_')[1]}.json")))["best"]["cfg"]


def best_epoch(rundir, s, seed):
    """Epoch of the lowest early-stopping loss in the ATTN_BO run (run length minus patience)."""
    ep = json.load(open(os.path.join(rundir, "final", f"ATTN_BO_S{s}_s{seed}.json")))["epochs"]
    return ep - 6 if ep < 60 else ep


def run_subject(tag, s, rundir, seeds):
    seeds = [sd for sd in seeds if not os.path.exists(os.path.join(rundir, "final", f"{tag}_S{s}_s{sd}.json"))]
    if not seeds:
        return
    out = os.path.join(rundir, "final", f"{tag}_S{s}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    X, y, r = load_subject(s)
    fit, es, te = mask(r, FIT_REPS), mask(r, ES_REPS), mask(r, TEST_REPS)
    sc = Scaler().fit(X[fit | es]); Xs = sc(X)
    cfg = config(tag, rundir)
    for seed in seeds:
        t = time.time()
        if tag == "ATTN_BO_FULL":
            allr = fit | es
            m, ep = train(KIND[tag], cfg, Xs[allr], y[allr], None, None, seed=seed, fixed_epochs=best_epoch(rundir, s, seed))
        else:
            m, ep = train(KIND[tag], cfg, Xs[fit], y[fit], Xs[es], y[es], seed=seed)
        sec = time.time() - t
        z = logits(m, Xs[te])
        rec = dict(seed=seed, epochs=ep, train_sec=sec, params=n_params(m), **metrics(z, y[te]))
        if seed == 0:
            torch.save(m.state_dict(), os.path.join(rundir, "final", f"{tag}_S{s}_seed0.pt"))
        base = os.path.join(rundir, "final", f"{tag}_S{s}_s{seed}")
        np.savez_compressed(base + ".npz", z=z.astype(np.float16), y=y[te],
                            z_es=logits(m, Xs[es]).astype(np.float16), y_es=y[es])
        json.dump(dict(tag=tag, subject=s, cfg=cfg, **rec), open(base + ".json", "w"), indent=1)
        print(tag, s, seed, ep, round(sec), round(rec["acc"], 4), flush=True)


if __name__ == "__main__":
    tag, rundir = sys.argv[1], sys.argv[2]
    subs = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] != "all" else range(1, 28)
    seeds = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0]
    for s in subs:
        run_subject(tag, s, rundir, seeds)
