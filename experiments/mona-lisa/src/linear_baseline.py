"""Reference only (NOT part of the Mona Lisa model): a *trained* linear classifier on raw pixels."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from scipy.optimize import minimize
from mnist import load
Xtr, ytr, Xte, yte = load()
Xb = np.hstack([Xtr, np.ones((len(Xtr), 1))]); Y = np.eye(10)[ytr]; lam = 1e-4
def f(w):
    W = w.reshape(785, 10); Z = Xb@W; Z -= Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    loss = -np.log(P[np.arange(len(ytr)), ytr] + 1e-12).mean() + lam*(W**2).sum()
    return loss, (Xb.T@(P - Y)/len(ytr) + 2*lam*W).ravel()
w = minimize(f, np.zeros(7850), jac=True, method='L-BFGS-B', options={'maxiter': 300}).x.reshape(785, 10)
print('linear (multinomial logistic regression) test accuracy:', ((np.hstack([Xte, np.ones((len(Xte), 1))])@w).argmax(1) == yte).mean())
