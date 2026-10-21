"""Ridge encoding: model features -> per-voxel linear ridge, solved in feature space.

Features are centred (or z-scored) with train-image statistics, voxels are z-scored with train
statistics, and each voxel picks its alpha from ALPHAS by 5-fold CV on the train images. The score
is Pearson r between predicted and actual held-out responses.

The model is the linear-kernel ridge of himalaya's KernelRidgeCV(kernel="precomputed", cv=5) on gram(X), which
this module used until 2026-10-20: same alphas on the same Gram scale, same CV folds (KFold(5), no shuffle), same
selection (mean negative MSE over folds, ties -> larger alpha), no intercept. It is solved from the D x D Z'Z in
float64 (or the full-rank n x n ZZ' when D > n) instead of the float32 eigendecomposition of the (n, n) Gram: with many
more images than features (TVSD: 17.8k-22k images, 512-d GAP) that Gram is rank D, and float32 eigh turns its
n - D zero eigenvalues into noise of either sign, which spoils small alphas (TVSD looped d16 IT, iter16: test r
0.42 instead of 0.63). The feature-space solve never forms the n x n Gram and is much faster for D << n.
"""
import numpy as np
import torch

ALPHAS = np.logspace(-10, 10, 20)
N_FOLDS = 5
# Voxels solved at once; lower it to bound GPU memory (results are identical). N_ALPHAS_BATCH is unused (kept for
# configs written for the himalaya solver).
N_ALPHAS_BATCH, N_TARGETS_BATCH = 5, 1000


def design(X, train_idx, device, *, features):
    """(n, D) float32 design Z on device, scaled so that Z @ Z.T == gram(X, train_idx, device, features=features).

    features="zscore": every feature z-scored, then divided by sqrt(D) (equal weight per block).
    features="center": features only centred (as in the Gallant lab voxelwise tutorials, StandardScaler(with_std=False)),
        so units keep their relative scales; Z is divided by the root mean squared train norm (the Gram's mean train
        diagonal). That is one global scale factor, the same as rescaling the alpha grid, so the per-voxel CV over
        alphas works on a comparable range for every model.
    """
    Z = X.to(device).float()
    tr = torch.as_tensor(train_idx, device=Z.device)
    Z = Z - Z[tr].mean(0)
    if features == "zscore":
        return Z / (Z[tr].std(0) + 1e-8) / Z.shape[1] ** 0.5
    if features == "center":
        return Z / Z[tr].pow(2).sum(1).mean().sqrt().clamp_min(1e-6)
    return Z / Z.shape[1] ** 0.5


def gram(X, train_idx, device, chunk=20_000, *, features):
    """Linear Gram of X with train-image statistics (see design); chunked over features, so it also fits raw
    flattened activations."""
    n, D = X.shape
    K = torch.zeros(n, n, dtype=torch.float32, device=device)
    for j in range(0, D, chunk):
        xc = X[:, j:j + chunk].to(device).float()
        xc = xc - xc[train_idx].mean(0)
        if features == "zscore":
            xc = xc / (xc[train_idx].std(0) + 1e-8)
        K += xc @ xc.T
    if features == "center":
        return K / K.diagonal()[train_idx].mean().clamp_min(1e-12)
    return K / D


def zscore_targets(Y, tr, te):
    """Y[tr] and Y[te], each z-scored per voxel with the train mean and std."""
    mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-8
    return (Y[tr] - mu) / sd, (Y[te] - mu) / sd


def pearson(a, b):
    """Per-column Pearson r of two (n, t) tensors."""
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-12)


class _Ridge:
    """Ridge of Y on Z for every alpha at once: coef(a, cols) such that the prediction is (Zq @ V) @ coef,
    i.e. Zq (Z'Z + a I)^-1 Z'Y.

    Only the eigendecomposition of the smaller Gram is done in float64: D <= n, Z'Z (D x D; prediction
    Zq V diag(1/(lam + a)) V'Z'Y); D > n, ZZ' (n x n, full rank; prediction Zq Z'U diag(1/(lam + a)) U'Y). Neither
    has structural zero eigenvalues, and float64 keeps the small ones far below any alpha even at condition numbers
    ~1e17. The projections of Y stay in the input dtype: their rounding error only enters along directions whose
    prediction weight is equally small."""

    def __init__(self, Z, Y):
        self.dtype = Z.dtype
        Zd = Z.double()
        if Z.shape[1] <= Z.shape[0]:
            lam, V = torch.linalg.eigh(Zd.T @ Zd)
            self.V = V.to(Z.dtype)
            self.PtY = (Z @ self.V).T @ Y  # V'Z'Y
        else:
            lam, U = torch.linalg.eigh(Zd @ Zd.T)
            U = U.to(Z.dtype)
            self.V, self.PtY = Z.T @ U, U.T @ Y  # Z'U, U'Y
        self.lam, self.PtY = lam.clamp_min(0), self.PtY.double()

    def coef(self, alpha, cols=slice(None)):
        return (self.PtY[:, cols] / (self.lam + alpha)[:, None]).to(self.dtype)

    def predict(self, Zq, alpha, cols=slice(None)):
        return (Zq @ self.V) @ self.coef(alpha, cols)


def _fit_predict(Z, Y, tr, te):
    """Per-voxel ridge fit on tr with alpha by 5-fold CV; returns (chosen alphas, z-scored Y[te], its prediction)."""
    dev = Z.device
    tr_t, te_t = torch.as_tensor(tr, device=dev), torch.as_tensor(te, device=dev)
    Ytr, Yte = zscore_targets(Y.to(dev), tr_t, te_t)
    Ztr, t = Z[tr_t], Ytr.shape[1]
    cv =torch.zeros(len(ALPHAS), t, dtype=torch.float64, device=dev)
    n = len(tr)
    for val in np.array_split(np.arange(n), N_FOLDS):  # = sklearn KFold(N_FOLDS), as himalaya's cv=5
        fit = torch.as_tensor(np.setdiff1d(np.arange(n), val), device=dev)
        val = torch.as_tensor(val, device=dev)
        r = _Ridge(Ztr[fit], Ytr[fit])
        Zv = Ztr[val] @ r.V
        for b in range(0, t, N_TARGETS_BATCH):
            cols = slice(b, b + N_TARGETS_BATCH)
            for i, a in enumerate(ALPHAS):
                pred = Zv @ r.coef(float(a), cols)
                cv[i, cols] -= ((Ytr[val, cols] - pred) ** 2).mean(0).double()
    cv = cv / N_FOLDS + torch.log(torch.as_tensor(ALPHAS, device=dev))[:, None] * 1e-10  # ties -> larger alpha
    best = cv.argmax(0)
    r = _Ridge(Ztr, Ytr)
    Zte = Z[te_t] @ r.V
    pred = torch.empty_like(Yte)
    for i in best.unique().tolist():
        cols = (best == i).nonzero().squeeze(1)
        pred[:, cols] = Zte @ r.coef(float(ALPHAS[i]), cols)
    return ALPHAS[best.cpu().numpy()], Yte, pred


def unit_scores(Z, Y, tr, te, subsets=None):
    """One ridge fit on rows tr of the design Z (see design): ((n_voxels,) Pearson r on te, chosen alphas,
    {name: (n_voxels,) r on each subset of te (positions into te)}), float64 numpy. Voxels are fit independently,
    so any set of them can share the fit."""
    a, Yte, pred = _fit_predict(Z, Y, tr, te)
    r = lambda x, y: pearson(x, y).double().cpu().numpy()
    pick = lambda x, v: x[torch.as_tensor(v, device=x.device)]
    return r(Yte, pred), a, {k: r(pick(Yte, v), pick(pred, v)) for k, v in (subsets or {}).items()}


def voxel_scores(Z, Y, tr, te):
    """(n_voxels,) float32 Pearson r on te of a ridge fit on rows tr of the design Z."""
    return unit_scores(Z, Y, tr, te)[0].astype(np.float32)


def fit_val_split(n):
    """Fixed 80/20 split of n train images into (fit, val) indices, RandomState(42)."""
    perm = np.random.RandomState(42).permutation(n)
    return perm[: int(0.8 * n)], perm[int(0.8 * n):]


def subject_targets(resp, pos, device="cuda"):
    """(feature rows of the train then test images, their responses Y on device, n train) of one region x subject;
    pos: image id -> feature row."""
    tr_ids, te_ids = list(resp["train"]), list(resp["test"])
    idx = [pos[str(i)] for i in tr_ids + te_ids]
    Y = torch.as_tensor(np.stack([resp["train"][i] for i in tr_ids] + [resp["test"][i] for i in te_ids]),
                        dtype=torch.float32, device=device)
    return idx, Y, len(tr_ids)
