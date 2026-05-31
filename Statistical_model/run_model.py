#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from tqdm import tqdm

from factor_utils import (
    exponential_weights, sort_svd,
    align_svd_to_previous, estimate_current_idio_vols,
)


def estimate_current_model(returns, t, gamma, B_prev, window_corr, corr_half_life, n_factors):
    assert window_corr - 1 <= t < returns.shape[0] - 1

    if gamma == 0:
        idio_vols = np.ones(returns.shape[1])
    else:
        idio_vols = estimate_current_idio_vols(returns, t)

    scale = idio_vols ** gamma
    W_sigma = np.diag(1 / scale)
    W_time = np.diag(exponential_weights(window_corr, corr_half_life))

    R_window = returns.iloc[t - window_corr + 1:t + 1, :].T.values
    R_scaled = W_sigma @ R_window @ W_time

    U, S, Vt = svds(R_scaled, k=n_factors)
    U, S, Vt = sort_svd(U, S, Vt)
    B_ranked = np.diag(scale) @ U
    U, S, Vt = align_svd_to_previous(U, S, Vt, match_current=B_ranked, match_prev=B_prev)

    B = np.diag(scale) @ U
    next_returns_window = returns.iloc[t - window_corr + 2:t + 2, :].T.values
    next_return = next_returns_window[:, -1]
    f_t = (U.T @ W_sigma @ next_returns_window)[:, -1]
    resid = next_return - B @ f_t

    return {"B": B, "U": U, "f_t": f_t, "resid": resid, "idio_vols": idio_vols}


def run(data_path, output_dir, gamma, n_factors, window_corr, corr_half_life):
    print(f"Loading data from {data_path}")
    returns = pd.read_hdf(data_path)
    returns = returns.apply(pd.to_numeric).sort_index()

    start_t = window_corr - 1
    end_t = returns.shape[0] - 1

    print(
        f"gamma={gamma}  n_factors={n_factors}  window={window_corr}  "
        f"dates={returns.index[start_t].date()} → {returns.index[end_t].date()}"
    )

    models = {}
    B_prev = None
    prev_f_t = None
    prev_resid = None

    for t in tqdm(range(start_t, end_t)):
        models[t] = {"f_t": prev_f_t, "resid": prev_resid}
        current = estimate_current_model(
            returns, t, gamma, B_prev, window_corr, corr_half_life, n_factors
        )
        models[t].update({k: current[k] for k in ["B", "U", "idio_vols"]})
        prev_f_t = current["f_t"]
        prev_resid = current["resid"]
        B_prev = current["B"]

    factor_index = returns.index[start_t + 1:end_t]
    model_index = returns.index[start_t:end_t]
    tickers = returns.columns
    factor_cols = [f"Factor {i}" for i in range(1, n_factors + 1)]
    carac_cols = [f"Carac {i}" for i in range(1, n_factors + 1)]

    factors = pd.DataFrame(
        np.array([models[t]["f_t"] for t in range(start_t + 1, end_t)]),
        index=factor_index,
        columns=factor_cols,
    )
    carac_idx = pd.MultiIndex.from_product(
        [list(model_index), list(tickers)], names=["Date", "Tickers"]
    )
    characteristics = pd.DataFrame(
        np.vstack([models[t]["B"] for t in range(start_t, end_t)]),
        index=carac_idx,
        columns=carac_cols,
    )
    residuals = pd.DataFrame(
        np.array([models[t]["resid"] for t in range(start_t + 1, end_t)]),
        index=factor_index,
        columns=tickers,
    )

    characteristics *= 100
    factors *= 0.01

    os.makedirs(output_dir, exist_ok=True)
    suffix = str(gamma).replace(".", "p")
    characteristics.to_parquet(f"{output_dir}/characteristics_gamma_{suffix}.parquet")
    residuals.to_parquet(f"{output_dir}/residuals_gamma_{suffix}.parquet")
    factors.to_parquet(f"{output_dir}/factors_gamma_{suffix}.parquet")
    print(f"Saved outputs to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rolling PCA factor model")
    parser.add_argument("--data-path", default="/data/daily_returns_SPX.h5")
    parser.add_argument("--output-dir", default="/data/outputs")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Idio-vol scaling: 0=none, 1=full")
    parser.add_argument("--n-factors", type=int, default=10)
    parser.add_argument("--window-corr", type=int, default=800)
    parser.add_argument("--corr-half-life", type=int, default=200)
    args = parser.parse_args()

    run(
        data_path=args.data_path,
        output_dir=args.output_dir,
        gamma=args.gamma,
        n_factors=args.n_factors,
        window_corr=args.window_corr,
        corr_half_life=args.corr_half_life,
    )
