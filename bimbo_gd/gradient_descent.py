"""Linear Regression optimised with Gradient Descent - pure NumPy.

Model:     y_hat = X @ w + b
Loss:      J(w, b) = 1/(2n) * sum((y_hat - y)^2) + (l2/2) * ||w||^2      (MSE + Ridge)
Gradient:  dJ/dw = (1/n) * X^T (y_hat - y) + l2 * w
           dJ/db = (1/n) * sum(y_hat - y)

Three variants (they differ in how many rows are used per update):
    "batch"      : all n rows per step       -> stable, slow on large data
    "mini_batch" : batch_size rows per step  -> good trade-off (recommended)
    "sgd"        : 1 row per step            -> noisy, slow Python loop
"""
from __future__ import annotations

import numpy as np


class LinearRegressionGD:
    def __init__(self, method="mini_batch", learning_rate=0.05, n_epochs=50,
                 batch_size=4096, l2=0.0, tol=1e-6, patience=5,
                 random_state=42, verbose=True):
        assert method in ("batch", "mini_batch", "sgd")
        self.method, self.lr, self.n_epochs = method, learning_rate, n_epochs
        self.batch_size, self.l2, self.tol, self.patience = batch_size, l2, tol, patience
        self.random_state, self.verbose = random_state, verbose
        self.w = self.b = self.mu = self.sigma = None
        self.history = {"train_loss": [], "val_loss": []}

    # ---------- helpers ----------
    def _scale(self, X):
        return (X - self.mu) / self.sigma

    def _loss(self, Xs, y):
        err = Xs @ self.w + self.b - y
        return 0.5 * np.mean(err ** 2) + 0.5 * self.l2 * np.sum(self.w ** 2)

    def _bs(self, n):
        return {"batch": n, "sgd": 1}.get(self.method, min(self.batch_size, n))

    # ---------- training on one in-memory dataset ----------
    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype="float64"); y = np.asarray(y, dtype="float64")
        n, d = X.shape
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0); self.sigma[self.sigma == 0] = 1.0   # z-score scaling
        Xs = self._scale(X)
        has_val = X_val is not None
        if has_val:
            Xv = self._scale(np.asarray(X_val, dtype="float64"))
            yv = np.asarray(y_val, dtype="float64")

        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(d); self.b = float(y.mean())     # bias starts at mean(y) for faster convergence
        self.history = {"train_loss": [], "val_loss": []}
        bs, best, best_params, bad = self._bs(n), np.inf, None, 0

        for epoch in range(1, self.n_epochs + 1):
            idx = rng.permutation(n) if self.method != "batch" else np.arange(n)
            for s in range(0, n, bs):
                j = idx[s:s + bs]
                err = Xs[j] @ self.w + self.b - y[j]
                self.w -= self.lr * (Xs[j].T @ err / len(j) + self.l2 * self.w)
                self.b -= self.lr * err.mean()

            tr = self._loss(Xs, y); self.history["train_loss"].append(tr)
            monitor = tr
            if has_val:
                vl = self._loss(Xv, yv); self.history["val_loss"].append(vl); monitor = vl
            if self.verbose and (epoch == 1 or epoch % max(1, self.n_epochs // 10) == 0):
                msg = f"[{self.method}] epoch {epoch:>4}/{self.n_epochs}  train={tr:.5f}"
                print(msg + (f"  val={vl:.5f}" if has_val else ""))

            # early stopping
            if monitor < best - self.tol:
                best, best_params, bad = monitor, (self.w.copy(), self.b), 0
            else:
                bad += 1
                if bad >= self.patience:
                    if self.verbose: print(f"Early stopping at epoch {epoch}")
                    break
        if best_params is not None:
            self.w, self.b = best_params
        return self

    def predict(self, X):
        return self._scale(np.asarray(X, dtype="float64")) @ self.w + self.b

    # ---------- incremental training over big data, one batch at a time ----------
    _SIGMA_FLOOR = 0.05   # avoids dividing by a near-zero std for almost-constant features

    def init_scaler(self, X, y):
        """Initialise from the first batch. The scaler is then updated gradually on later batches."""
        X = np.asarray(X, dtype="float64")
        self._n, self._s1, self._s2 = len(X), X.sum(axis=0), (X ** 2).sum(axis=0)
        self.mu = self._s1 / self._n
        self.sigma = np.maximum(np.sqrt(np.maximum(self._s2 / self._n - self.mu ** 2, 0)), self._SIGMA_FLOOR)
        self.w = np.zeros(X.shape[1]); self.b = float(np.mean(y))
        self._rng = np.random.default_rng(self.random_state)
        self.history = {"train_loss": [], "val_loss": []}
        return self

    def _update_scaler(self, X):
        """Update the cumulative mean/std across batches and adjust w, b so the PREDICTION FUNCTION IS UNCHANGED."""
        self._n += len(X); self._s1 += X.sum(axis=0); self._s2 += (X ** 2).sum(axis=0)
        mu = self._s1 / self._n
        sigma = np.maximum(np.sqrt(np.maximum(self._s2 / self._n - mu ** 2, 0)), self._SIGMA_FLOOR)
        self.b += float(np.sum(self.w * (mu - self.mu) / self.sigma))
        self.w = self.w * sigma / self.sigma
        self.mu, self.sigma = mu, sigma

    def partial_fit(self, X, y, n_epochs=5):
        """Continue Gradient Descent on one batch, KEEPING the learned w, b (warm start)."""
        X = np.asarray(X, dtype="float64"); y = np.asarray(y, dtype="float64")
        self._update_scaler(X)
        Xs = self._scale(X)
        n = len(y); bs = self._bs(n)
        for _ in range(n_epochs):
            idx = self._rng.permutation(n) if self.method != "batch" else np.arange(n)
            for s in range(0, n, bs):
                j = idx[s:s + bs]
                err = Xs[j] @ self.w + self.b - y[j]
                self.w -= self.lr * (Xs[j].T @ err / len(j) + self.l2 * self.w)
                self.b -= self.lr * err.mean()
            loss = self._loss(Xs, y)
            if not np.isfinite(loss) or loss > 1e6:
                raise FloatingPointError("Loss diverged - lower the learning rate (e.g. 0.01).")
            self.history["train_loss"].append(loss)
        return self.history["train_loss"][-1]

    def raw_coefficients(self):
        """Convert weights back to the original feature scale:  y = intercept + sum(coef_i * x_i)."""
        coef = self.w / self.sigma
        return float(self.b - np.sum(self.w * self.mu / self.sigma)), coef

    # ---------- save / load ----------
    def save(self, path, feature_names):
        np.savez(path, w=self.w, b=self.b, mu=self.mu, sigma=self.sigma,
                 features=np.array(feature_names))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        m = cls(verbose=False)
        m.w, m.b, m.mu, m.sigma = z["w"], float(z["b"]), z["mu"], z["sigma"]
        return m, list(z["features"])
