"""NinaPro DB1 windowing.

DB1 stores the Otto Bock electrode output, which is already rectified and RMS-smoothed
and sampled at 100 Hz, so no band-pass filtering or rectification is applied here.
Each subject's three exercises are concatenated; movement labels of exercises 2 and 3 are
offset by 12 and 29 so that the 52 movements plus rest form 53 classes.  Windows of
W = 20 samples (200 ms) are taken every 10 samples (100 ms) and kept only when the
re-labelled stimulus ("restimulus") and repetition are constant inside the window.
Rest windows are thinned to one in four to limit class imbalance.

usage: python prep_db1.py RAW_DIR OUT_DIR
"""
import os, sys, json
import numpy as np
import scipy.io as sio

W, STEP, REST_KEEP = 20, 10, 4
OFFSET = {1: 0, 2: 12, 3: 29}


def subject(raw, s):
    E, Y, R = [], [], []
    for e in (1, 2, 3):
        m = sio.loadmat(os.path.join(raw, f"S{s}_A1_E{e}.mat"))
        y = m["restimulus"].ravel().astype(int)
        y = np.where(y > 0, y + OFFSET[e], 0)
        E.append(m["emg"].astype(np.float32)); Y.append(y); R.append(m["rerepetition"].ravel().astype(int))
    emg, y, rep = np.concatenate(E), np.concatenate(Y), np.concatenate(R)
    Xw, yw, rw = [], [], []
    rest_count = 0
    for st in range(0, len(y) - W + 1, STEP):
        yy, rr = y[st:st + W], rep[st:st + W]
        if yy.min() != yy.max() or rr.min() != rr.max():
            continue
        lab = int(yy[0])
        if lab == 0:
            rest_count += 1
            if rest_count % REST_KEEP:
                continue
        Xw.append(emg[st:st + W]); yw.append(lab); rw.append(int(rr[0]))
    return np.stack(Xw), np.array(yw), np.array(rw)


def main(raw, out):
    os.makedirs(out, exist_ok=True)
    info = {}
    for s in range(1, 28):
        X, y, r = subject(raw, s)
        # rest windows carry repetition 0; they are assigned to repetitions by position so
        # that the repetition split also separates rest periods in time
        if (r == 0).any():
            idx0 = np.where(r == 0)[0]
            nz = np.where(r > 0)[0]
            nearest = nz[np.clip(np.searchsorted(nz, idx0), 0, len(nz) - 1)]
            r[idx0] = r[nearest]
        np.savez_compressed(os.path.join(out, f"S{s}.npz"), X=X, y=y, rep=r)
        info[s] = dict(windows=int(len(y)), classes=int(len(np.unique(y))), rest=int((y == 0).sum()))
        print(s, info[s], flush=True)
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
