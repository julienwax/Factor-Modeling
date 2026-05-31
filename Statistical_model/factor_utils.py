import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import svds


def exponential_weights(window, half_life):
    tau = half_life / np.log(2)
    kappa = np.sqrt(window * (1 - np.exp(-2 / tau)) / (1 - np.exp(-2 * window / tau)))
    return kappa * np.exp(-np.arange(window)[::-1] / tau)


def sort_svd(U, S, Vt):
    idx = np.argsort(S)[::-1]
    return U[:, idx], S[idx], Vt[idx, :]


def normalize_columns(X):
    norms = np.linalg.norm(X, axis=0)
    if np.any(norms == 0):
        raise ValueError("Cannot match factors with a zero-norm column")
    return X / norms


def orient_svd_deterministic(U, S, Vt, match_current=None):
    """
    Resolve SVD sign ambiguity: force the largest-magnitude loading in each
    eigenvector to be positive. match_current overrides U as the basis for
    finding the anchor (use raw-return loadings B_t instead of standardized U).
    """
    U, S, Vt = U.copy(), S.copy(), Vt.copy()
    basis = U if match_current is None else match_current
    for j in range(U.shape[1]):
        anchor = np.argmax(np.abs(basis[:, j]))
        if U[anchor, j] < 0:
            U[:, j] *= -1
            Vt[j, :] *= -1
    return U, S, Vt


def align_svd_to_previous(U, S, Vt, match_current=None, match_prev=None):
    """
    Resolve SVD permutation + sign ambiguity across time steps.

    Matches each current eigenvector to the previous labeled one with maximum
    absolute dot product (Hungarian algorithm), then flips signs so matched
    dot products are positive. Falls back to orient_svd_deterministic on the
    first date when match_prev is None.
    """
    U, S, Vt = U.copy(), S.copy(), Vt.copy()
    k = U.shape[1]
    current_basis = U if match_current is None else match_current

    if match_prev is None:
        return orient_svd_deterministic(U, S, Vt, current_basis)

    if match_prev.shape != current_basis.shape:
        raise ValueError(
            f"Previous basis shape {match_prev.shape} != current shape {current_basis.shape}"
        )

    overlap = normalize_columns(match_prev).T @ normalize_columns(current_basis)
    rows, cols = linear_sum_assignment(-np.abs(overlap))

    U_out = np.empty_like(U)
    S_out = np.empty_like(S)
    Vt_out = np.empty_like(Vt)
    for prev_label, curr_rank in zip(rows, cols):
        sign = 1.0 if overlap[prev_label, curr_rank] >= 0 else -1.0
        U_out[:, prev_label] = sign * U[:, curr_rank]
        S_out[prev_label] = S[curr_rank]
        Vt_out[prev_label, :] = sign * Vt[curr_rank, :]

    return U_out, S_out, Vt_out


def estimate_current_idio_vols(returns, t, window_idio=600, idio_half_life=150, n_idio_factors=10):
    assert window_idio - 1 <= t < returns.shape[0]
    weights = exponential_weights(window_idio, idio_half_life)
    R_window = returns.iloc[t - window_idio + 1:t + 1, :].T.values
    R_weighted = R_window @ np.diag(weights)
    U, S, Vt = svds(R_weighted, k=n_idio_factors)
    U, S, Vt = sort_svd(U, S, Vt)
    residuals = R_weighted - U @ np.diag(S) @ Vt
    return np.maximum(np.sqrt(np.mean(residuals**2, axis=1)), 1e-8)


def l2_normalized_turnover(v_prev, v_curr):
    v_prev = np.asarray(v_prev).ravel()
    v_curr = np.asarray(v_curr).ravel()
    return np.sum(np.abs(v_curr / np.linalg.norm(v_curr) - v_prev / np.linalg.norm(v_prev)))


def l1_normalized_turnover(v_prev, v_curr):
    v_prev = np.asarray(v_prev).ravel()
    v_curr = np.asarray(v_curr).ravel()
    w_prev = v_prev / np.sum(np.abs(v_prev))
    w_curr = v_curr / np.sum(np.abs(v_curr))
    return 0.5 * np.sum(np.abs(w_curr - w_prev))


def cosine_turnover(v_prev, v_curr):
    v_prev = np.asarray(v_prev).ravel()
    v_curr = np.asarray(v_curr).ravel()
    cos = np.dot(v_prev, v_curr) / (np.linalg.norm(v_prev) * np.linalg.norm(v_curr))
    return 2 * (1 - abs(np.clip(cos, -1.0, 1.0)))
