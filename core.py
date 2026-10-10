"""Shared components for the NinaPro DB1 experiments: data access, models, training,
hyperparameter decoding, optimisers and metrics."""
import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
N_CLASSES = 53
EPS = 1e-3                       # floor added before the log transform (DB1 units are mV-scale RMS)
TRAIN_REPS = (1, 3, 4, 6, 8, 9, 10)
TEST_REPS = (2, 5, 7)
DATA = os.environ.get("DB1", os.path.join(os.path.dirname(os.path.abspath(__file__)), "db1"))


# ----------------------------------------------------------------- data
def load_subject(s):
    d = np.load(os.path.join(DATA, f"S{s}.npz"))
    return d["X"].astype(np.float32), d["y"].astype(np.int64), d["rep"].astype(np.int64)


def log_tf(X):
    return np.log(np.maximum(X, 0) + EPS).astype(np.float32)


class Scaler:
    """Channel-wise z-score in the log domain, fitted on training windows only."""
    def fit(self, X):
        Z = log_tf(X).reshape(-1, X.shape[2])
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        return self

    def __call__(self, X):
        return ((log_tf(X) - self.mu) / self.sd).astype(np.float32)


def mask(rep, reps):
    return np.isin(rep, reps)


# ----------------------------------------------------------------- search space
# Every optimiser moves in u in [0,1]^7; decode() maps a point to a configuration.
DIM = 7


def decode(u):
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    return dict(hidden=int(round(16 + u[0] * 112)),        # 16 .. 128 units per direction
                layers=1 if u[1] < 0.5 else 2,
                dropout=float(0.5 * u[2]),
                lr=float(10 ** (-3.5 + 1.5 * u[3])),       # 3.2e-4 .. 1e-2
                wd=float(10 ** (-6 + 3 * u[4])),           # 1e-6 .. 1e-3
                att=int(round(8 + u[5] * 56)),             # attention width 8 .. 64
                aug=float(0.4 * u[6]))                     # log-gain augmentation std 0 .. 0.4


# Manually chosen configuration (the "manual tuning" reference of the study)
DEFAULT = dict(hidden=64, layers=1, dropout=0.2, lr=1e-3, wd=1e-4, att=32, aug=0.2)


# ----------------------------------------------------------------- networks
class BiLSTM(nn.Module):
    """Bi-LSTM whose summary is the concatenated final forward and backward states."""
    def __init__(self, d_in, hidden, layers, dropout, bidirectional=True, **_):
        super().__init__()
        self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        self.bi = bidirectional
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear((2 if bidirectional else 1) * hidden, N_CLASSES)

    def forward(self, x, return_att=False):
        _, (h, _) = self.rnn(x)
        z = torch.cat([h[-2], h[-1]], 1) if self.bi else h[-1]
        out = self.head(self.drop(z))
        return (out, None) if return_att else out


class AttBiLSTM(nn.Module):
    """Bi-LSTM followed by additive attention pooling over time steps."""
    def __init__(self, d_in, hidden, layers, dropout, att=32, **_):
        super().__init__()
        self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0, bidirectional=True)
        self.W = nn.Linear(2 * hidden, att)
        self.v = nn.Linear(att, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, N_CLASSES)

    def forward(self, x, return_att=False):
        H, _ = self.rnn(x)                                   # (B, T, 2h)
        e = self.v(torch.tanh(self.W(H))).squeeze(-1)        # (B, T)
        a = torch.softmax(e, 1)
        c = torch.bmm(a.unsqueeze(1), H).squeeze(1)          # (B, 2h)
        out = self.head(self.drop(c))
        return (out, a) if return_att else out


class TCN(nn.Module):
    """Three-block dilated temporal convolutional baseline with global average pooling."""
    def __init__(self, d_in, hidden=64, dropout=0.2, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, hidden, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4), nn.ReLU(), nn.BatchNorm1d(hidden))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, N_CLASSES)

    def forward(self, x, return_att=False):
        out = self.head(self.drop(self.net(x.transpose(1, 2)).mean(2)))
        return (out, None) if return_att else out


def build(kind, cfg, d_in):
    if kind == "bilstm":
        return BiLSTM(d_in, **cfg)
    if kind == "attn":
        return AttBiLSTM(d_in, **cfg)
    if kind == "tcn":
        return TCN(d_in, cfg["hidden"], cfg["dropout"])
    raise ValueError(kind)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------- training
def class_weights(y):
    c = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = c.sum() / (N_CLASSES * np.maximum(c, 1))
    return torch.tensor(w, dtype=torch.float32)


def augment(xb, g, n, gen):
    """Per-window, per-channel log-gain offset (std g) plus white noise (std n), both in the
    standardised log domain."""
    if g <= 0 and n <= 0:
        return xb
    B, T, C = xb.shape
    off = torch.randn(B, 1, C, generator=gen) * g
    return xb + off + torch.randn(B, T, C, generator=gen) * n


def train(kind, cfg, Xtr, ytr, Xva, yva, seed, max_epochs=60, patience=6, batch=256, fixed_epochs=None):
    """Adam with class-balanced cross-entropy, gain/noise augmentation, gradient clipping
    and early stopping on the validation loss; returns the best model (eval mode) and the
    number of epochs run.  With fixed_epochs=E the model is trained for exactly E epochs on
    the training data and the final weights are returned (no validation set is used)."""
    aug = (cfg.get("aug", 0.0), cfg.get("aug", 0.0) / 4)
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(kind, cfg, Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    w = class_weights(ytr)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xv, yv = (torch.from_numpy(Xva), torch.from_numpy(yva)) if Xva is not None else (None, None)
    best, state, wait, ep = 1e9, None, 0, 0
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    for ep in range(1, (fixed_epochs or max_epochs) + 1):
        m.train()
        idx = rng.permutation(len(Xt))
        for b in range(0, len(idx), batch):
            j = idx[b:b + batch]
            loss = F.cross_entropy(m(augment(Xt[j], aug[0], aug[1], gen)), yt[j], weight=w)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        m.eval()
        if fixed_epochs:
            continue
        with torch.no_grad():
            vl = F.cross_entropy(torch.cat([m(Xv[i:i + 2048]) for i in range(0, len(Xv), 2048)]), yv, weight=w).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if not fixed_epochs:
        m.load_state_dict(state)
    m.eval()
    return m, ep


@torch.no_grad()
def logits(m, X, batch=2048, return_att=False):
    Z, A = [], []
    for i in range(0, len(X), batch):
        z, a = m(torch.from_numpy(X[i:i + batch]), return_att=True)
        Z.append(z); A.append(a)
    Z = torch.cat(Z).numpy()
    if return_att:
        return Z, (torch.cat(A).numpy() if A[0] is not None else None)
    return Z


# ----------------------------------------------------------------- hand-crafted features
def features(X, is_log=False):
    """Per-channel mean, standard deviation, waveform length, maximum and slope of the
    log-RMS envelope inside the window (5 x 10 = 50 features).  With is_log=True the input
    is already in the (standardised) log domain."""
    L = X if is_log else log_tf(X)
    mean, sd = L.mean(1), L.std(1)
    wl = np.abs(np.diff(L, axis=1)).sum(1)
    mx = L.max(1)
    t = np.arange(L.shape[1]) - (L.shape[1] - 1) / 2
    slope = (L * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.concatenate([mean, sd, wl, mx, slope], 1).astype(np.float32)


# ----------------------------------------------------------------- metrics
def balanced_accuracy(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def macro_f1(y, p):
    f = []
    for c in np.unique(y):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(prob, y, bins=15):
    conf = prob.max(1); acc = (prob.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(acc[s].mean() - conf[s].mean())
    return float(e)


def nll(prob, y):
    return float(-np.mean(np.log(prob[np.arange(len(y)), y] + 1e-12)))


def metrics(z, y):
    p = z.argmax(1); pr = softmax(z)
    mv = y > 0
    return dict(acc=float(np.mean(p == y)), ba=balanced_accuracy(y, p), f1=macro_f1(y, p),
                acc_mov=float(np.mean(p[mv] == y[mv])), ece=ece(pr, y), nll=nll(pr, y))


# ----------------------------------------------------------------- optimisers
class RandomSearch:
    name = "RS"

    def __init__(self, dim, budget, rng):
        self.d, self.B, self.rng = dim, budget, rng

    def run(self, f):
        return [f(self.rng.random(self.d)) for _ in range(self.B)]


class BayesOpt:
    """Gaussian-process Bayesian optimisation (Optuna GPSampler, log expected improvement)."""
    name = "BO"

    def __init__(self, dim, budget, seed, n_startup=6, init=None):
        self.d, self.B, self.seed, self.ns, self.init = dim, budget, seed, n_startup, init

    def run(self, f):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.GPSampler(seed=self.seed, n_startup_trials=self.ns)
        st = optuna.create_study(direction="maximize", sampler=sampler)
        if self.init is not None:                       # optional shared initial design
            for u in self.init:
                st.enqueue_trial({f"u{k}": float(u[k]) for k in range(self.d)})
        trace = []

        def obj(tr):
            u = np.array([tr.suggest_float(f"u{k}", 0.0, 1.0) for k in range(self.d)])
            v = f(u); trace.append(v); return v
        st.optimize(obj, n_trials=self.B)
        return trace
