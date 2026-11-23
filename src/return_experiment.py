"""Return / directional forecasting experiment on top of the baseline.

Trains a PyTorch ``BaselineGRU`` (or ``BaselineLSTM``) on next-step return
targets (log-return or percentage), reconstructs implied prices via
``P_hat_t = P_{t-1} * exp(r_hat_t)``, and evaluates return/directional and
price-level metrics side-by-side with naive persistence and the raw price-level
GRU on the *exact same* chronological split and seed.

* Strict chronological ``70/15/15`` split (configurable); feature scalers are
  fitted on the training split only (zero future leakage); return targets are
  derived from raw price pairs within each split so no out-of-split
  information is used.
* Losses: ``mse`` (default, on return targets) or ``hybrid`` (MSE on return
  magnitude + BCE on the direction sign, via a two-head GRU). A pure-BCE mode
  is intentionally omitted because it cannot produce a price magnitude for the
  reconstruction comparison.
* Frictionless diagnostics (clearly not trading advice): sign-strategy hit
  rate and cumulative return of a zero-fee signal-ignoring-fees strategy vs.
  buy-and-hold, computed on the same out-of-sample returns.

Usage (from the repository root)::

    python -m src.return_experiment --help
    python -m src.return_experiment --sample-synthetic --seed 42
    python -m src.return_experiment --ticker AAPL --seed 42
    python -m src.return_experiment --ticker AAPL --return-type pct --loss hybrid
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # PyTorch is optional at import time (needed only for model work).
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment without torch
    TORCH_AVAILABLE = False

from src.baseline import (  # noqa: E402
    DataConfig,
    ModelConfig,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_LOOKBACK,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    DEFAULT_VAL_FRAC,
    build_model,
    compute_metrics,
    evaluate_model,
    evaluate_persistence,
    make_run_id,
    prepare_dataframe,
    require_torch,
    resolve_data_source,
    resolve_device,
    resolve_sentiment_columns,
    save_metrics_json,
    set_seed,
    train_network,
)

PRICE_METRIC_KEYS = ("mae", "rmse", "nrmse", "mape", "r2", "directional_accuracy")
RETURN_METRIC_KEYS = (
    "directional_accuracy",
    "return_mae",
    "return_rmse",
    "hit_rate_sign_strategy",
)


@dataclass
class ReturnExperimentConfig:
    """Settings for one return-experiment run."""

    lookback: int = DEFAULT_LOOKBACK
    val_frac: float = DEFAULT_VAL_FRAC
    test_frac: float = DEFAULT_TEST_FRAC
    return_type: str = "log"  # "log" | "pct" | "logvol" | "pctvol"
    loss: str = "mse"  # "mse" | "hybrid"
    deadzone_eps: float = 0.005  # |r| threshold for strict directional accuracy
    baseline: str = "none"  # "none" | "ridge"
    seed: int = DEFAULT_SEED
    out_dir: Path = Path("results")


# ---------------------------------------------------------------------------
# Return targets / reconstruction / metrics
# ---------------------------------------------------------------------------

def _window_return_targets(window: Any, return_type: str) -> np.ndarray:
    """Next-step return targets per window, from raw price pairs.

    ``window.y_raw[:, 0]`` is ``P_t`` and ``window.prev_raw`` is ``P_{t-1}``,
    so targets are computed strictly within the split segment.
    """
    p_prev = window.prev_raw
    p_now = window.y_raw[:, 0]
    if np.any(p_now <= 0.0) or np.any(p_prev <= 0.0):
        raise ValueError(
            "Return targets require strictly positive prices "
            "(found non-positive values in the data)."
        )
    if return_type == "log":
        return np.log(p_now / p_prev)
    return (p_now / p_prev - 1.0) * 100.0


def _reconstruct_prices(
    prev_raw: np.ndarray, pred_return: np.ndarray, return_type: str
) -> np.ndarray:
    """``P_hat_t = P_{t-1} * exp(r_hat_t)`` (or percentage equivalent)."""
    pred_return = np.asarray(pred_return, dtype=float).ravel()
    if return_type == "log":
        return prev_raw * np.exp(pred_return)
    return prev_raw * (1.0 + pred_return / 100.0)


def compute_return_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    return_type: str = "log",
    *,
    deadzone_eps: float = 0.005,
) -> dict[str, float]:
    """Return-space metrics on natural return units.

    * directional_accuracy - sign(actual return) == sign(predicted return);
      zero-magnitude predictions only count if the actual move is also zero.
    * directional_accuracy_strict - sign match *excluding* steps whose actual
      move is inside the dead zone (|r| <= eps); NaN if no qualifying step.
    * deadzone_share - fraction of steps inside the dead zone (small moves).
    * return_mae / return_rmse - error in return units.
    * hit_rate_sign_strategy - same as directional accuracy (zero-fee).
    * cumulative_return_buy_and_hold / cumulative_return_sign_strategy -
      compounded multi-period returns, decimal. The sign strategy allocates
      +1/−1/0 on the predicted sign and collects the *actual* return; labeled
      as a frictionless diagnostic, not trading advice.
    """
    actual = np.asarray(actual, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if len(actual) != len(pred):
        raise ValueError(
            f"Length mismatch: actual={len(actual)}, pred={len(pred)}"
        )
    err = actual - pred
    return_mae = float(np.mean(np.abs(err)))
    return_rmse = float(np.sqrt(np.mean(err**2)))

    act_dir = np.sign(actual)
    pred_dir = np.sign(pred)
    directional = float(np.mean(act_dir == pred_dir))

    # Dead-zone (thresholded) directional accuracy on meaningful moves.
    meaningful = np.abs(actual) > deadzone_eps
    n_meaningful = int(np.sum(meaningful))
    if n_meaningful > 0:
        strict = float(np.mean(act_dir[meaningful] == pred_dir[meaningful]))
    else:
        strict = float("nan")
    deadzone_share = float(np.mean(~meaningful))

    # Decimal returns for compounding (log-sum-exp form: stable, no overflow).
    r_dec = np.expm1(actual) if return_type == "log" else actual / 100.0
    strat = pred_dir * r_dec  # sign allocation x actual return
    buy_hold = float(np.expm1(np.sum(np.log1p(r_dec))))
    sign_strategy = float(np.expm1(np.sum(np.log1p(strat))))

    return {
        "directional_accuracy": directional,
        "directional_accuracy_strict": strict,
        "deadzone_share": deadzone_share,
        "deadzone_eps": deadzone_eps,
        "return_mae": return_mae,
        "return_rmse": return_rmse,
        "hit_rate_sign_strategy": directional,
        "cumulative_return_buy_and_hold": buy_hold,
        "cumulative_return_sign_strategy": sign_strategy,
    }


# ---------------------------------------------------------------------------
# Hybrid (MSE + BCE direction) model and training
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class HybridGRU(nn.Module):
        """GRU with two heads: return magnitude (linear) + direction logit."""

        def __init__(
            self,
            input_size: int,
            hidden_size: int = DEFAULT_HIDDEN,
            num_layers: int = 1,
        ) -> None:
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.ret_head = nn.Linear(hidden_size, 1)
            self.dir_head = nn.Linear(hidden_size, 1)

        def forward(self, x: Any) -> tuple[Any, Any]:
            out, _ = self.gru(x)
            h = out[:, -1, :]
            return self.ret_head(h), self.dir_head(h)

else:  # pragma: no cover - placeholder when torch is missing

    HybridGRU = None  # type: ignore[assignment, misc]


def _train_return_model(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ModelConfig,
    loss_mode: str = "mse",
) -> dict[str, Any]:
    """Train a return-target model with MSE (or hybrid MSE+BCE) + early stop."""
    require_torch()
    device = resolve_device(cfg.device)
    model = model.to(device)

    # Targets are pre-flattened to 1-D so losses never broadcast (B,1) vs (B,).
    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(np.asarray(y_train, dtype=np.float32).reshape(-1)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(np.asarray(y_val, dtype=np.float32).reshape(-1)),
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def batch_loss(out: Any, yb: Any) -> Any:
        """Loss on flattened 1-D targets (matches ``(B,)`` head slices exactly)."""
        if loss_mode == "hybrid":
            out_ret, out_dir = out
            return F.mse_loss(out_ret[:, 0], yb) + 0.5 * F.binary_cross_entropy_with_logits(
                out_dir[:, 0], (yb >= 0).float()
            )
        if loss_mode == "dir_margin":
            # Directional-margin surrogate: Huber magnitude + BCE direction,
            # with k calibrated on the train-split return std.
            k = 1.0 / max(float(np.std(y_train)), 1e-4)
            magnitude = F.huber_loss(out[:, 0], yb, delta=1.0)
            direction = F.binary_cross_entropy_with_logits(
                k * out[:, 0], (yb > 0).float()
            )
            return 0.5 * magnitude + 0.5 * direction
        return F.mse_loss(out[:, 0], yb)

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = batch_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_ds)

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vtotal += batch_loss(model(xb), yb).item() * xb.size(0)
        val_loss = vtotal / len(val_ds)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience and epoch >= 2:
                break

    train_time = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "epochs_run": len(history),
        "train_time_s": train_time,
    }


def _predict_returns(
    model: Any,
    x: np.ndarray,
    *,
    batch_size: int = 256,
    device: str = "auto",
    hybrid: bool = False,
) -> np.ndarray:
    """Return-head predictions (numpy in/out)."""
    require_torch()
    device = resolve_device(device)
    model = model.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i : i + batch_size]).to(device)
            out = model(batch)
            if hybrid:
                out = out[0]
            outputs.append(out.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _fit_ridge_baseline(prep: Any, ret_train_raw: np.ndarray,
                        ret_test_raw: np.ndarray, base_type: str, *,
                        deadzone_eps: float) -> dict[str, Any]:
    """Simple ridge baseline on the same leak-free windows (return targets)."""
    from sklearn.linear_model import Ridge
    x_train = prep.train.X.reshape(prep.train.X.shape[0], -1)
    x_test = prep.test.X.reshape(prep.test.X.shape[0], -1)
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(x_train, ret_train_raw)
    pred_ret = ridge.predict(x_test).reshape(-1, 1)
    pred_price = _reconstruct_prices(prep.test.prev_raw, pred_ret, base_type)
    return {
        "model": "Ridge(alpha=1.0)",
        "price_metrics_reconstructed": compute_metrics(
            prep.test.y_raw, pred_price, prep.test.prev_raw
        ),
        "return_metrics": compute_return_metrics(
            ret_test_raw, pred_ret, base_type, deadzone_eps=deadzone_eps
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration + reporting
# ---------------------------------------------------------------------------

def run_return_experiment(
    df: pd.DataFrame,
    *,
    data_cfg: DataConfig,
    exp_cfg: ReturnExperimentConfig,
    model_cfg: ModelConfig,
    source_label: str,
) -> dict[str, Any]:
    """Run persistence vs. raw-price GRU vs. return-target GRU on one split."""
    require_torch()
    set_seed(model_cfg.seed)

    # Canonical chronological split; scaler fit on train rows only.
    prep = prepare_dataframe(
        df,
        data_cfg=data_cfg,
        lookback=exp_cfg.lookback,
        horizon=1,
        val_frac=exp_cfg.val_frac,
        test_frac=exp_cfg.test_frac,
        scaler_kind="minmax",
    )

    # 1) Naive persistence (price level).
    per = evaluate_persistence(prep)

    # 2) Raw price-level GRU (same split/seed).
    raw_model = build_model(model_cfg.model, prep.input_size, model_cfg)
    raw_hist = train_network(
        raw_model, prep.train.X, prep.train.y_scaled, prep.val.X, prep.val.y_scaled,
        model_cfg,
    )
    raw_ev = evaluate_model(raw_model, prep, model_cfg)

    # 3) Return-target GRU (same split/seed, same input windows).
    # Variance-standardized targets ("logvol"/"pctvol") divide by the TRAIN-split
    # return std only (leak-free); predictions are rescaled before reconstruction.
    return_type = exp_cfg.return_type
    vol = return_type.endswith("vol")
    base_type = return_type[:-3] if vol else return_type
    ret_train_raw = _window_return_targets(prep.train, base_type)
    ret_val_raw = _window_return_targets(prep.val, base_type)
    ret_test_raw = _window_return_targets(prep.test, base_type)
    if vol:
        sigma = float(np.std(ret_train_raw))
        if sigma < 1e-12:
            raise ValueError("Train-split return std is ~0; cannot standardize targets.")
        ret_train, ret_val, ret_test = (
            ret_train_raw / sigma, ret_val_raw / sigma, ret_test_raw / sigma
        )
    else:
        sigma = 1.0
        ret_train, ret_val, ret_test = ret_train_raw, ret_val_raw, ret_test_raw

    hybrid = exp_cfg.loss == "hybrid"
    if hybrid:
        ret_model = HybridGRU(
            prep.input_size,
            hidden_size=model_cfg.hidden_size,
            num_layers=model_cfg.num_layers or 1,
        )
    else:
        ret_model = build_model(model_cfg.model, prep.input_size, model_cfg)
    ret_hist = _train_return_model(
        ret_model, prep.train.X, ret_train, prep.val.X, ret_val,
        model_cfg, loss_mode=exp_cfg.loss,
    )
    t0 = time.perf_counter()
    pred_ret_scaled = _predict_returns(
        ret_model, prep.test.X,
        batch_size=model_cfg.batch_size, device=model_cfg.device, hybrid=hybrid,
    )
    ret_infer_time = time.perf_counter() - t0
    pred_ret = pred_ret_scaled * sigma
    pred_price = _reconstruct_prices(prep.test.prev_raw, pred_ret, base_type)
    price_metrics = compute_metrics(prep.test.y_raw, pred_price, prep.test.prev_raw)
    return_metrics = compute_return_metrics(
        ret_test_raw, pred_ret, base_type, deadzone_eps=exp_cfg.deadzone_eps
    )

    # Optional simple linear (ridge) baseline on the same leak-free split.
    ridge_out = None
    if exp_cfg.baseline == "ridge":
        ridge_out = _fit_ridge_baseline(
            prep, ret_train_raw, ret_test_raw, base_type,
            deadzone_eps=exp_cfg.deadzone_eps,
        )

    results: dict[str, Any] = {
        "persistence": {
            "price_metrics": per["metrics"],
            "infer_time_s": per["infer_time_s"],
        },
        "raw_price_gru": {
            "price_metrics": raw_ev["metrics"],
            "best_epoch": raw_hist["best_epoch"],
            "epochs_run": raw_hist["epochs_run"],
            "best_val_loss": raw_hist["best_val_loss"],
            "train_time_s": raw_hist["train_time_s"],
            "infer_time_s": raw_ev["infer_time_s"],
        },
        "return_gru": {
            "target": f"{exp_cfg.return_type}_return",
            "loss": exp_cfg.loss,
            "price_metrics_reconstructed": price_metrics,
            "return_metrics": return_metrics,
            "best_epoch": ret_hist["best_epoch"],
            "epochs_run": ret_hist["epochs_run"],
            "best_val_loss": ret_hist["best_val_loss"],
            "train_time_s": ret_hist["train_time_s"],
            "infer_time_s": ret_infer_time,
            "vol_standardized": vol,
            "train_std": float(sigma),
            "diagnostics_note": (
                "Sign-strategy cumulative return is a zero-fee, frictionless "
                "diagnostic computed on out-of-sample returns; it is not "
                "trading advice (no costs, slippage, or risk controls)."
            ),
        },
    }
    if ridge_out is not None:
        results["ridge_baseline"] = ridge_out
    run_id = make_run_id("return_experiment")
    payload: dict[str, Any] = {
        "run_id": run_id,
        "project": "improving-gru-sentiment-stock-forecasting-padmakumari-maam",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_source": source_label,
        "n_rows": len(df),
        "config": {
            "experiment": asdict(exp_cfg),
            "model": asdict(model_cfg),
            "data": asdict(data_cfg),
        },
        "metric_keys": {
            "price": list(PRICE_METRIC_KEYS),
            "return": list(RETURN_METRIC_KEYS),
        },
        "results": results,
    }

    # Split ranges (recompute proportionally to match prepare_dataframe).
    n = len(df)
    n_test = max(1, round(n * exp_cfg.test_frac))
    n_val = max(1, round(n * exp_cfg.val_frac))
    n_train = n - n_val - n_test
    payload["split"] = {
        "train_range": [0, n_train],
        "val_range": [n_train, n_train + n_val],
        "test_range": [n_train + n_val, n],
    }

    json_path = save_metrics_json(exp_cfg.out_dir, run_id, payload)
    md_path = _write_markdown(exp_cfg.out_dir, run_id, payload)
    payload["saved"] = {"json": str(json_path), "markdown": str(md_path)}
    save_metrics_json(exp_cfg.out_dir, run_id, payload)

    _print_summary(payload)
    return payload


def _write_markdown(out_dir: Path, run_id: str, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.md"
    cfg = payload["config"]
    split = payload["split"]
    lines = [
        f"# Return experiment report — `{run_id}`",
        "",
        f"- **Data source:** {payload['data_source']} ({payload['n_rows']} rows)",
        f"- **Split:** chronological {cfg['experiment']['lookback']}-based windows; "
        f"train={split['train_range']}, val={split['val_range']}, test={split['test_range']}",
        f"- **Model:** {cfg['model']['model']} (hidden={cfg['model']['hidden_size']}, "
        f"seed={cfg['model']['seed']})",
        f"- **Return target:** {cfg['experiment']['return_type']} return; "
        f"loss={cfg['experiment']['loss']}",
        "",
    ]

    res = payload["results"]

    lines += ["## Price-level comparison (same split & seed)", ""]
    lines += [
        "| model | MAE | RMSE | NRMSE | MAPE | R2 | DirAcc |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = [
        ("persistence", res["persistence"]["price_metrics"]),
        ("raw price GRU", res["raw_price_gru"]["price_metrics"]),
        ("return GRU (reconstructed)", res["return_gru"]["price_metrics_reconstructed"]),
    ]
    for label, m in rows:
        lines.append(
            f"| {label} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['nrmse']:.4f} | "
            f"{m['mape']:.3f}% | {m['r2']:.4f} | {m['directional_accuracy']:.4f} |"
        )
    lines += [""]

    lines += ["## Return-space metrics (return GRU)", ""]
    rm = res["return_gru"]["return_metrics"]
    lines += [
        f"- **directional_accuracy:** {rm['directional_accuracy']:.4f}",
        f"- **directional_accuracy_strict (|r| > {rm['deadzone_eps']:.3f}):** "
        f"{rm['directional_accuracy_strict']:.4f}  (deadzone share {rm['deadzone_share']:.3f})",
        f"- **return_mae:** {rm['return_mae']:.4f}",
        f"- **return_rmse:** {rm['return_rmse']:.4f}",
        f"- **hit_rate_sign_strategy:** {rm['hit_rate_sign_strategy']:.4f}",
        f"- **cumulative_return_buy_and_hold:** {rm['cumulative_return_buy_and_hold']:.4%}",
        f"- **cumulative_return_sign_strategy:** {rm['cumulative_return_sign_strategy']:.4%}",
        "",
        f"_Note:_ {res['return_gru']['diagnostics_note']}",
        "",
    ]
    if "ridge_baseline" in res:
        rr = res["ridge_baseline"]["return_metrics"]
        rp = res["ridge_baseline"]["price_metrics_reconstructed"]
        lines += [
            "### Ridge baseline (same split & seed)", "",
            f"- Model: {res['ridge_baseline']['model']}",
            f"- Price MAE {rp['mae']:.4f} · RMSE {rp['rmse']:.4f} · "
            f"nRMSE {rp['nrmse']:.4f} · MAPE {rp['mape']:.3f}%",
            f"- DirAcc {rr['directional_accuracy']:.4f} · strict "
            f"{rr['directional_accuracy_strict']:.4f} · deadzone {rr['deadzone_share']:.3f}",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _print_summary(payload: dict[str, Any]) -> None:
    cfg = payload["config"]
    res = payload["results"]
    print("\n" + "=" * 72)
    print("RETURN EXPERIMENT SUMMARY")
    print("=" * 72)
    print(f"run_id      : {payload['run_id']}")
    print(f"data_source : {payload['data_source']}  ({payload['n_rows']} rows)")
    print(f"seed        : {cfg['model']['seed']}  lookback={cfg['experiment']['lookback']}  "
          f"target={cfg['experiment']['return_type']}  loss={cfg['experiment']['loss']}")

    print("\n-- Price-level comparison (same split & seed) --")
    print(f"{'model':26s} {'MAE':>8s} {'RMSE':>8s} {'NRMSE':>8s} {'MAPE':>8s} {'R2':>7s} {'DirAcc':>7s}")
    rows = [
        ("persistence", res["persistence"]["price_metrics"]),
        ("raw price GRU", res["raw_price_gru"]["price_metrics"]),
        ("return GRU (reconstructed)", res["return_gru"]["price_metrics_reconstructed"]),
    ]
    for label, m in rows:
        print(f"{label:26s} {m['mae']:8.4f} {m['rmse']:8.4f} {m['nrmse']:8.4f} "
              f"{m['mape']:7.3f}% {m['r2']:7.4f} {m['directional_accuracy']:7.4f}")

    rm = res["return_gru"]["return_metrics"]
    print("\n-- Return-space metrics (return GRU) --")
    print(f"   directional_accuracy={rm['directional_accuracy']:.4f}  "
          f"strict_dir_acc(|r|>{rm['deadzone_eps']:.3f})={rm['directional_accuracy_strict']:.4f}  "
          f"deadzone_share={rm['deadzone_share']:.3f}")
    print(f"   return_mae={rm['return_mae']:.4f}  return_rmse={rm['return_rmse']:.4f}")
    print(f"   hit_rate_sign_strategy={rm['hit_rate_sign_strategy']:.4f}  "
          f"buy&hold_cum={rm['cumulative_return_buy_and_hold']:.4%}  "
          f"sign_strategy_cum={rm['cumulative_return_sign_strategy']:.4%}")
    if "ridge_baseline" in res:
        rr = res["ridge_baseline"]["return_metrics"]
        rp = res["ridge_baseline"]["price_metrics_reconstructed"]
        print(f"   [ridge] MAE={rp['mae']:.4f} RMSE={rp['rmse']:.4f} "
              f"DirAcc={rr['directional_accuracy']:.4f} "
              f"strict={rr['directional_accuracy_strict']:.4f}")
    print(f"   [{res['return_gru']['diagnostics_note']}]")

    print("\nsaved:")
    for key, val in payload["saved"].items():
        print(f"   {key}: {val}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.return_experiment",
        description=(
            "Return/directional forecasting vs. raw price-level GRU and "
            "persistence on the same chronological split and seed."
        ),
        epilog=(
            "Examples:\n"
            "  python -m src.return_experiment --sample-synthetic --seed 42\n"
            "  python -m src.return_experiment --ticker AAPL --seed 42\n"
            "  python -m src.return_experiment --ticker AAPL --return-type pct --loss hybrid\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data = parser.add_argument_group("data source (exactly one required)")
    data.add_argument("--csv", metavar="FILE", help="path to a price CSV")
    data.add_argument("--ticker", metavar="SYMBOL", help="Yahoo Finance ticker (yfinance)")
    data.add_argument("--sample-synthetic", action="store_true",
                      help="generate a small synthetic series for testing")
    data.add_argument("--date-col", default="Date")
    data.add_argument("--close-col", default="Close")
    data.add_argument("--sentiment-cols", default="", metavar="COLS",
                      help="comma-separated sentiment columns (default: auto-detect)")
    data.add_argument("--no-sentiment", action="store_true",
                      help="disable sentiment features entirely (price-only)")

    run = parser.add_argument_group("experiment settings")
    run.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    run.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    run.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    run.add_argument("--return-type", choices=["log", "pct", "logvol", "pctvol"],
                     default="log",
                     help="return target: log/pct (natural) or logvol/pctvol "
                          "(variance-standardized by train-split std)")
    run.add_argument("--loss", choices=["mse", "hybrid", "dir_margin"], default="mse",
                     help="mse (default), hybrid MSE+BCE, or directional-margin "
                          "(Huber + BCE, k calibrated on train return std)")
    run.add_argument("--eps-deadzone", type=float, default=0.005,
                     help="|r| dead-zone threshold for strict directional accuracy (default 0.005)")
    run.add_argument("--baseline", choices=["none", "ridge"], default="none",
                     help="extra simple baseline: ridge regression on same split (default none)")
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)

    mod = parser.add_argument_group("model / training")
    mod.add_argument("--model", choices=["gru", "lstm"], default="gru")
    mod.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN)
    mod.add_argument("--layers", type=int, default=None)
    mod.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    mod.add_argument("--lr", type=float, default=DEFAULT_LR)
    mod.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    mod.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    mod.add_argument("--device", default="auto", help="auto | cpu | cuda")
    mod.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.no_sentiment:
        sentiment_cols: tuple[str, ...] = ()
    else:
        sentiment_cols = tuple(
            c.strip() for c in args.sentiment_cols.split(",") if c.strip()
        )

    df, source_label = resolve_data_source(
        path=args.csv,
        ticker=args.ticker,
        synthetic=args.sample_synthetic,
        seed=args.seed,
        date_col=args.date_col,
        close_col=args.close_col,
        sentiment_cols=sentiment_cols,
    )

    if not args.no_sentiment and not sentiment_cols:
        sentiment_cols = resolve_sentiment_columns(
            df, date_col=args.date_col, close_col=args.close_col
        )
        if sentiment_cols:
            print(f"[data] auto-detected sentiment column(s): {list(sentiment_cols)}")

    if sentiment_cols:
        missing = [c for c in sentiment_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"Sentiment column(s) not found in data: {missing}")

    data_cfg = DataConfig(
        date_col=args.date_col,
        close_col=args.close_col,
        sentiment_cols=sentiment_cols,
    )
    model_cfg = ModelConfig(
        seed=args.seed,
        model=args.model,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        dropout=0.0,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        batch_size=args.batch_size,
        device=args.device,
    )
    exp_cfg = ReturnExperimentConfig(
        lookback=args.lookback,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        return_type=args.return_type,
        loss=args.loss,
        deadzone_eps=args.eps_deadzone,
        baseline=args.baseline,
        seed=args.seed,
        out_dir=args.out_dir,
    )

    try:
        run_return_experiment(
            df,
            data_cfg=data_cfg,
            exp_cfg=exp_cfg,
            model_cfg=model_cfg,
            source_label=source_label,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
