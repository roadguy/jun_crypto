"""XGBoost Model A의 누수 방지 Walk-Forward 및 포트폴리오 진단.

운영 예측에서는 호출하지 않는다. 개발 Walk-Forward와 최종 Lockbox는
분리하며, Lockbox는 저장 결과가 존재하면 기본적으로 다시 평가하지 않는다.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import crypto_xgb_core as core


LOCKBOX_DAYS = 90
MIN_RUIN_TRADES = 300
BOOTSTRAP_REPEATS = 300
NOISE_REPEATS = 3
DIAGNOSTIC_VERSION = "portfolio_diagnostics_v1"


def _safe_metrics(y_true, probability) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "log_loss": np.nan, "brier": np.nan}
    return core.probability_metrics(y, p)


def split_development_lockbox(
    model_df: pd.DataFrame,
    lockbox_days: int = LOCKBOX_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """가장 최근 lockbox_days를 개발 과정에서 완전히 분리한다."""
    ordered = model_df.sort_values("timestamp").reset_index(drop=True)
    cutoff = ordered["timestamp"].iloc[-1] - pd.Timedelta(days=lockbox_days)
    lockbox_start = int(ordered["timestamp"].searchsorted(cutoff, side="left"))
    development_end = lockbox_start - core.PURGE_BARS
    development = ordered.iloc[:development_end].copy()
    lockbox = ordered.iloc[lockbox_start:].copy()
    if len(development) < 400 or len(lockbox) < 30:
        raise ValueError(
            "최종 Lockbox를 분리할 데이터가 부족합니다. "
            f"개발 {len(development):,}개, Lockbox {len(lockbox):,}개입니다."
        )
    metadata = {
        "lockbox_days": int(lockbox_days),
        "development_start": str(development["timestamp"].iloc[0]),
        "development_end": str(development["timestamp"].iloc[-1]),
        "lockbox_start": str(lockbox["timestamp"].iloc[0]),
        "lockbox_end": str(lockbox["timestamp"].iloc[-1]),
        "lockbox_samples": len(lockbox),
        "purge_bars": int(core.PURGE_BARS),
    }
    return development, lockbox, metadata


def build_fold_schedule(
    model_df: pd.DataFrame,
    timeframe: str,
) -> list[tuple[int, int, int, int]]:
    config = core.timeframe_config(timeframe)
    first_time = model_df["timestamp"].iloc[0]
    final_time = model_df["timestamp"].iloc[-1]
    test_start_time = first_time + pd.Timedelta(days=config.initial_train_days)
    schedule = []
    while test_start_time <= final_time:
        test_end_time = min(
            test_start_time + pd.Timedelta(days=config.test_days),
            final_time + config.delta,
        )
        test_start = int(model_df["timestamp"].searchsorted(test_start_time, side="left"))
        test_end = int(model_df["timestamp"].searchsorted(test_end_time, side="left"))
        train_end = test_start - core.PURGE_BARS
        if test_start >= len(model_df):
            break
        if train_end >= 400 and test_end - test_start >= 30:
            schedule.append((0, train_end, test_start, test_end))
        test_start_time += pd.Timedelta(days=config.step_days)
    return schedule


def calibration_reliability(y_true, probability) -> tuple[pd.DataFrame, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    bins = min(10, max(2, len(y) // 100))
    actual, predicted = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    table = pd.DataFrame(
        {"mean_probability": predicted, "actual_up_ratio": actual}
    )
    table["gap"] = table["actual_up_ratio"] - table["mean_probability"]
    return table, float(table["gap"].abs().mean())


def _baseline_probabilities(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """각 Fold의 과거 데이터만 사용해 고정 베이스라인 확률을 만든다."""
    prior = float(train["target_up"].mean())
    constant = np.full(len(test), prior, dtype=float)
    train_up = train["return_1"] > 0
    up_rate = float(train.loc[train_up, "target_up"].mean()) if train_up.any() else prior
    down_rate = float(train.loc[~train_up, "target_up"].mean()) if (~train_up).any() else prior
    momentum = np.where(test["return_1"].to_numpy() > 0, up_rate, down_rate)
    logistic = Pipeline(
        [
            ("scale", RobustScaler()),
            ("model", LogisticRegression(C=0.01, l1_ratio=0.0, max_iter=2000,
                                          random_state=core.RANDOM_STATE)),
        ]
    )
    logistic.fit(train[core.FEATURE_COLUMNS], train["target_up"].astype(int))
    logistic_probability = logistic.predict_proba(test[core.FEATURE_COLUMNS])[:, 1]
    return {
        "constant_probability_up": np.clip(constant, 1e-6, 1 - 1e-6),
        "momentum_probability_up": np.clip(momentum, 1e-6, 1 - 1e-6),
        "logistic_probability_up": np.clip(logistic_probability, 1e-6, 1 - 1e-6),
    }


def _permutation_rows(model, outer_test: pd.DataFrame, fold: int) -> list[dict[str, Any]]:
    """OOS AUC 감소량으로 한 번씩 섞은 피처 중요도를 계산한다."""
    y = outer_test["target_up"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return []
    x = outer_test[core.FEATURE_COLUMNS].copy()
    baseline = roc_auc_score(y, model.predict_proba(x)[:, 1])
    rng = np.random.default_rng(core.RANDOM_STATE + fold)
    rows = []
    for feature in core.FEATURE_COLUMNS:
        shuffled = x.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        permuted_auc = roc_auc_score(y, model.predict_proba(shuffled)[:, 1])
        rows.append({
            "fold": fold,
            "feature": feature,
            "raw_oos_auc": float(baseline),
            "permuted_auc": float(permuted_auc),
            "importance_auc_drop": float(baseline - permuted_auc),
        })
    return rows


def _noise_feature_rows(
    pre_calibration: pd.DataFrame,
    outer_test: pd.DataFrame,
    config: core.XGBConfig,
    fold: int,
) -> list[dict[str, Any]]:
    """난수 피처가 중요도를 얻거나 OOS를 우연히 개선하는지 반복 확인한다."""
    rows = []
    columns = list(core.FEATURE_COLUMNS)
    for repeat in range(1, NOISE_REPEATS + 1):
        rng = np.random.default_rng(core.RANDOM_STATE + fold * 100 + repeat)
        train_x = pre_calibration[columns].copy()
        test_x = outer_test[columns].copy()
        train_x["random_noise"] = rng.normal(size=len(train_x))
        test_x["random_noise"] = rng.normal(size=len(test_x))
        model = core.make_model(config, early_stopping=False)
        model.fit(train_x, pre_calibration["target_up"].astype(int))
        probability = model.predict_proba(test_x)[:, 1]
        auc = _safe_metrics(outer_test["target_up"], probability)["roc_auc"]
        importance = np.asarray(model.feature_importances_, dtype=float)
        rank = int((-importance).argsort().tolist().index(len(importance) - 1) + 1)
        rows.append({
            "fold": fold,
            "repeat": repeat,
            "noise_oos_auc": float(auc),
            "noise_importance": float(importance[-1]),
            "noise_importance_rank": rank,
            "feature_count": len(importance),
            "noise_in_top_20pct": bool(rank <= max(1, int(np.ceil(len(importance) * 0.2)))),
        })
    return rows


def run_fold(
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    fold: int,
    run_noise_test: bool = False,
):
    fit_df, tune_df, calibration_fit, calibration_select = core.split_selection_data(outer_train)
    config, search = core.tune_model(fit_df, tune_df, fold=fold)
    pre_calibration = pd.concat([fit_df, tune_df], ignore_index=True)
    model = core.make_model(config, early_stopping=False)
    model.fit(pre_calibration[core.FEATURE_COLUMNS], pre_calibration["target_up"].astype(int))

    p_fit_raw = model.predict_proba(calibration_fit[core.FEATURE_COLUMNS])[:, 1]
    p_select_raw = model.predict_proba(calibration_select[core.FEATURE_COLUMNS])[:, 1]
    calibrators = core.fit_calibrators(calibration_fit["target_up"], p_fit_raw)
    selected, calibration_search = core.select_calibrator(
        calibrators, calibration_select["target_up"], p_select_raw
    )
    p_select = core.apply_calibrator(selected["name"], selected["calibrator"], p_select_raw)
    threshold, threshold_search = core.choose_threshold(
        p_select, calibration_select["future_return"]
    )
    p_train_raw = model.predict_proba(pre_calibration[core.FEATURE_COLUMNS])[:, 1]
    p_train = core.apply_calibrator(selected["name"], selected["calibrator"], p_train_raw)
    p_test_raw = model.predict_proba(outer_test[core.FEATURE_COLUMNS])[:, 1]
    p_test = core.apply_calibrator(selected["name"], selected["calibrator"], p_test_raw)
    train_metrics = _safe_metrics(pre_calibration["target_up"], p_train)
    validation_metrics = _safe_metrics(calibration_select["target_up"], p_select)
    oos_metrics = _safe_metrics(outer_test["target_up"], p_test)
    train_raw_metrics = _safe_metrics(pre_calibration["target_up"], p_train_raw)
    validation_raw_metrics = _safe_metrics(calibration_select["target_up"], p_select_raw)
    oos_raw_metrics = _safe_metrics(outer_test["target_up"], p_test_raw)
    prediction = (p_test >= 0.5).astype(int)

    frame = outer_test[["timestamp", "close", "future_return", "target_up"]].copy()
    frame["fold"] = fold
    frame["config"] = config.name
    frame["best_trees"] = config.n_estimators
    frame["calibration"] = selected["name"]
    frame["threshold"] = threshold
    frame["raw_probability_up"] = p_test_raw
    frame["probability_up"] = p_test
    frame["probability_down"] = 1 - p_test
    frame["prediction_up"] = prediction
    for column, values in _baseline_probabilities(outer_train, outer_test).items():
        frame[column] = values

    summary = {
        "fold": fold,
        "train_start": str(outer_train["timestamp"].iloc[0]),
        "train_end": str(outer_train["timestamp"].iloc[-1]),
        "test_start": str(outer_test["timestamp"].iloc[0]),
        "test_end": str(outer_test["timestamp"].iloc[-1]),
        "samples": len(outer_test),
        "config": config.name,
        "best_trees": config.n_estimators,
        "calibration": selected["name"],
        "threshold": threshold,
        "train_roc_auc": train_metrics["roc_auc"],
        "train_log_loss": train_metrics["log_loss"],
        "validation_roc_auc": validation_metrics["roc_auc"],
        "validation_log_loss": validation_metrics["log_loss"],
        "oos_roc_auc": oos_metrics["roc_auc"],
        "oos_log_loss": oos_metrics["log_loss"],
        "train_oos_auc_gap": train_metrics["roc_auc"] - oos_metrics["roc_auc"],
        "train_raw_roc_auc": train_raw_metrics["roc_auc"],
        "train_raw_log_loss": train_raw_metrics["log_loss"],
        "validation_raw_roc_auc": validation_raw_metrics["roc_auc"],
        "validation_raw_log_loss": validation_raw_metrics["log_loss"],
        "oos_raw_roc_auc": oos_raw_metrics["roc_auc"],
        "oos_raw_log_loss": oos_raw_metrics["log_loss"],
        "train_oos_raw_auc_gap": train_raw_metrics["roc_auc"] - oos_raw_metrics["roc_auc"],
        "roc_auc": oos_metrics["roc_auc"],
        "log_loss": oos_metrics["log_loss"],
        "brier": oos_metrics["brier"],
        "accuracy": float(accuracy_score(outer_test["target_up"].astype(int), prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(outer_test["target_up"].astype(int), prediction)
        ),
    }
    diagnostics = {
        "calibration": calibration_search,
        "threshold": threshold_search,
        "permutation": _permutation_rows(model, outer_test, fold),
        "noise": (_noise_feature_rows(pre_calibration, outer_test, config, fold)
                  if run_noise_test else []),
    }
    return frame, summary, search, diagnostics


def _simulate_fixed_position_strategy(
    predictions: pd.DataFrame,
    timeframe: str,
) -> dict[str, Any]:
    """신호가 유지되는 동안 계약 수량을 고정한 비용 반영 시뮬레이션."""
    ordered = predictions.sort_values("timestamp").reset_index(drop=True)
    position = core.positions_from_probability(
        ordered["probability_up"], ordered["threshold"]
    ).astype(int)
    close = ordered["close"].to_numpy(dtype=float)
    future_return = ordered["future_return"].to_numpy(dtype=float)
    timestamps = pd.to_datetime(ordered["timestamp"])
    delta = core.timeframe_config(timeframe).delta
    leverage = core.LEVERAGE * core.CAPITAL_FRACTION
    side_cost_rate = core.ONE_WAY_FEE_RATE + core.SLIPPAGE_RATE
    equity = 1.0
    equity_curve = np.full(len(ordered), np.nan)
    trade_returns: list[float] = []
    total_fee_amount = 0.0
    index = 0
    while index < len(ordered):
        direction = int(position[index])
        if direction == 0:
            equity_curve[index] = equity
            index += 1
            continue
        end = index + 1
        while end < len(ordered):
            contiguous = timestamps.iloc[end] == timestamps.iloc[end - 1] + delta
            if int(position[end]) != direction or not contiguous:
                break
            end += 1
        equity_before = equity
        entry_price = close[index]
        entry_notional = equity_before * leverage
        quantity = entry_notional / entry_price
        entry_fee = entry_notional * side_cost_rate
        total_fee_amount += entry_fee
        for bar in range(index, end):
            exit_mark = close[bar] * (1 + future_return[bar])
            marked_equity = equity_before - entry_fee + direction * quantity * (exit_mark - entry_price)
            if bar == end - 1:
                exit_fee = quantity * exit_mark * side_cost_rate
                total_fee_amount += exit_fee
                marked_equity -= exit_fee
            equity_curve[bar] = max(0.0, marked_equity)
        equity = float(equity_curve[end - 1])
        trade_returns.append(equity / equity_before - 1 if equity_before > 0 else -1.0)
        index = end
    equity_series = pd.Series(equity_curve).ffill().fillna(1.0).to_numpy(dtype=float)
    previous_equity = np.r_[1.0, equity_series[:-1]]
    bar_returns = np.divide(equity_series, previous_equity,
                            out=np.ones_like(equity_series),
                            where=previous_equity != 0) - 1
    peak = np.maximum.accumulate(np.r_[1.0, equity_series])
    drawdown = np.r_[1.0, equity_series] / peak - 1
    return {
        "position": position,
        "bar_returns": bar_returns,
        "equity": equity_series,
        "drawdown": drawdown[1:],
        "trade_returns": np.asarray(trade_returns, dtype=float),
        "total_fee": float(total_fee_amount),
        "bankrupt": bool(np.any(equity_series <= 0)),
    }


def _max_loss_streak(values) -> int:
    maximum = current = 0
    for value in values:
        if value < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _ruin_risk(trade_returns) -> dict[str, Any]:
    values = np.asarray(trade_returns, dtype=float)
    if len(values) < MIN_RUIN_TRADES:
        return {"status": "INSUFFICIENT_DATA", "available_trades": len(values),
                "required_trades": MIN_RUIN_TRADES}
    rng = np.random.default_rng(core.RANDOM_STATE)
    half = drawdown_30 = 0
    for _ in range(5000):
        sampled = rng.choice(values, size=300, replace=True)
        equity = np.cumprod(np.maximum(0, 1 + sampled))
        path = np.r_[1.0, equity]
        dd = path / np.maximum.accumulate(path) - 1
        half += int(np.min(equity) <= 0.5)
        drawdown_30 += int(np.min(dd) <= -0.30)
    return {"status": "AVAILABLE", "simulations": 5000, "horizon_trades": 300,
            "probability_capital_below_50pct": half / 5000,
            "probability_mdd_below_minus_30pct": drawdown_30 / 5000}


def strategy_summary(predictions: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    simulation = _simulate_fixed_position_strategy(predictions, timeframe)
    predictions["position"] = simulation["position"]
    predictions["net_strategy_return"] = simulation["bar_returns"]
    predictions["equity"] = simulation["equity"]
    predictions["drawdown"] = simulation["drawdown"]
    net = simulation["bar_returns"]
    trades = simulation["trade_returns"]
    periods = core.timeframe_config(timeframe).periods_per_year
    std = net.std(ddof=1)
    sharpe = net.mean() / std * np.sqrt(periods) if std > 0 else np.nan
    downside = net[net < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = net.mean() / downside_std * np.sqrt(periods) if np.isfinite(downside_std) and downside_std > 0 else np.nan
    wins, losses = trades[trades > 0], trades[trades < 0]
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    payoff = avg_win / abs(avg_loss) if len(wins) and len(losses) else np.nan
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) else np.nan
    return {
        "prediction_opportunities": len(predictions), "closed_trades": len(trades),
        "winning_trades": len(wins), "losing_trades": len(losses),
        "win_rate": float(len(wins) / len(trades)) if len(trades) else None,
        "average_win": float(avg_win) if np.isfinite(avg_win) else None,
        "average_loss": float(avg_loss) if np.isfinite(avg_loss) else None,
        "payoff_ratio": float(payoff) if np.isfinite(payoff) else None,
        "expectancy": float(trades.mean()) if len(trades) else None,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else None,
        "max_consecutive_losses": _max_loss_streak(trades),
        "cumulative_return": float(simulation["equity"][-1] - 1),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "sortino": float(sortino) if np.isfinite(sortino) else None,
        "mdd": float(np.min(simulation["drawdown"])),
        "total_fee": simulation["total_fee"],
        "exposure": float(np.mean(simulation["position"] != 0)),
        "bankrupt": simulation["bankrupt"],
        "execution_assumption": "신호 유지 중 계약 수량 고정, 신호 변경 시에만 청산·진입",
        "funding_included": False, "slippage_rate": float(core.SLIPPAGE_RATE),
        "ruin_risk": _ruin_risk(trades),
    }


def block_bootstrap_auc(y_true, probability, repeats: int = BOOTSTRAP_REPEATS) -> dict[str, Any]:
    """연속성을 일부 보존하는 moving-block bootstrap AUC 신뢰구간."""
    y, p = np.asarray(y_true, dtype=int), np.asarray(probability, dtype=float)
    n = len(y)
    block_length = max(5, int(round(n ** (1 / 3))))
    starts = np.arange(max(1, n - block_length + 1))
    blocks_needed = int(np.ceil(n / block_length))
    rng = np.random.default_rng(core.RANDOM_STATE)
    values = []
    for _ in range(repeats):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate([np.arange(s, min(s + block_length, n)) for s in chosen])[:n]
        if len(np.unique(y[indices])) >= 2:
            values.append(roc_auc_score(y[indices], p[indices]))
    if not values:
        return {"status": "UNAVAILABLE"}
    return {"status": "AVAILABLE", "method": "moving_block_bootstrap",
            "repeats": len(values), "block_length": block_length,
            "auc_ci_95_low": float(np.quantile(values, 0.025)),
            "auc_ci_95_high": float(np.quantile(values, 0.975))}


def _baseline_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    mapping = {"xgboost_model_a": "probability_up",
               "historical_rate": "constant_probability_up",
               "simple_momentum": "momentum_probability_up",
               "logistic_regression": "logistic_probability_up"}
    return {name: _safe_metrics(predictions["target_up"], predictions[column])
            for name, column in mapping.items()}


def _overfit_status(train_auc: float, oos_auc: float, fold_std: float) -> str:
    gap = train_auc - oos_auc
    if oos_auc <= 0.5:
        return "OOS 예측력 확인 안 됨"
    if gap >= 0.10 or fold_std >= 0.06:
        return "과적합 위험 높음"
    if gap >= 0.05 or fold_std >= 0.035:
        return "약한 과적합 가능성"
    return "뚜렷한 과적합 증거 없음"


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"],
                                   cwd=Path(__file__).resolve().parent,
                                   capture_output=True, text=True, check=True, timeout=2)
        return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def reproducibility_info(raw: pd.DataFrame, model_df: pd.DataFrame,
                         lockbox: dict[str, Any]) -> dict[str, Any]:
    return {
        "diagnostic_version": DIAGNOSTIC_VERSION, "model_version": core.MODEL_VERSION,
        "feature_version": core.MODEL_VERSION, "feature_columns": list(core.FEATURE_COLUMNS),
        "data_start": str(raw["timestamp"].iloc[0]), "data_end": str(raw["timestamp"].iloc[-1]),
        "raw_rows": len(raw), "model_rows": len(model_df),
        "python_version": sys.version.split()[0], "platform": platform.platform(),
        "pandas_version": pd.__version__, "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__, "random_seed": int(core.RANDOM_STATE),
        "generated_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
        "xgb_candidates": [vars(item) for item in core.XGB_CANDIDATES],
        "cost_assumptions": {"leverage": float(core.LEVERAGE),
                             "capital_fraction": float(core.CAPITAL_FRACTION),
                             "one_way_fee_rate": float(core.ONE_WAY_FEE_RATE),
                             "slippage_rate": float(core.SLIPPAGE_RATE),
                             "funding_included": False},
        "lockbox": lockbox,
    }


def _run_walk_forward_impl(symbol: str, timeframe: str, refresh: bool,
                           run_noise_test: bool) -> dict[str, Any]:
    raw, _, model_df = core.prepare_data(symbol, timeframe, refresh=refresh)
    development, _, lockbox_meta = split_development_lockbox(model_df)
    schedule = build_fold_schedule(development, timeframe)
    config = core.timeframe_config(timeframe)
    if not schedule:
        available_days = (development["timestamp"].iloc[-1] - development["timestamp"].iloc[0]).total_seconds() / 86400
        raise ValueError(f"Walk-Forward 구간을 만들 수 없습니다. 개발구간 약 {available_days:.0f}일, "
                         f"초기 학습 {config.initial_train_days}일과 최소 Test 표본이 필요합니다.")
    frames, fold_rows, searches, permutation_rows, noise_rows = [], [], [], [], []
    core.report("\n" + "=" * 76)
    core.report(f"{symbol} {timeframe} 개발구간 Expanding Walk-Forward")
    core.report("최종 90일 Lockbox는 학습·선택·검증에서 제외")
    core.report("=" * 76)
    for fold, (train_start, train_end, test_start, test_end) in enumerate(schedule, start=1):
        outer_train = development.iloc[train_start:train_end].copy()
        outer_test = development.iloc[test_start:test_end].copy()
        frame, fold_summary, search, diagnostics = run_fold(
            outer_train, outer_test, fold, run_noise_test=run_noise_test)
        frames.append(frame); fold_rows.append(fold_summary); searches.append(search)
        permutation_rows.extend(diagnostics["permutation"])
        noise_rows.extend(diagnostics["noise"])
        core.report(f"Fold {fold:02d} | {fold_summary['config']} | "
              f"{fold_summary['calibration']:8s} | t={fold_summary['threshold']:.2f} | "
              f"Raw Train={fold_summary['train_raw_roc_auc']:.4f} | "
              f"Raw OOS={fold_summary['oos_raw_roc_auc']:.4f} | "
              f"Cal OOS={fold_summary['oos_roc_auc']:.4f}")
    predictions = pd.concat(frames, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    search_df = pd.concat(searches, ignore_index=True)
    overall = _safe_metrics(predictions["target_up"], predictions["probability_up"])
    reliability, calibration_gap = calibration_reliability(predictions["target_up"],
                                                            predictions["probability_up"])
    strategy = strategy_summary(predictions, timeframe)
    train_auc = float(folds["train_raw_roc_auc"].mean())
    validation_auc = float(folds["validation_roc_auc"].mean())
    diagnostic_oos_auc = float(folds["oos_raw_roc_auc"].mean())
    fold_std = float(folds["oos_roc_auc"].std(ddof=1)) if len(folds) > 1 else 0.0
    paths = core.get_paths(symbol, timeframe)
    existing_lockbox = None
    if paths.lockbox_summary.exists():
        try:
            with paths.lockbox_summary.open(encoding="utf-8") as file:
                existing_lockbox = json.load(file)
        except (OSError, json.JSONDecodeError):
            pass
    lockbox_status = (existing_lockbox if existing_lockbox and
                      existing_lockbox.get("model_version") == core.MODEL_VERSION and
                      existing_lockbox.get("diagnostic_version") == DIAGNOSTIC_VERSION and
                      existing_lockbox.get("feature_columns") == list(core.FEATURE_COLUMNS)
                      else {"status": "RESERVED_NOT_EVALUATED", **lockbox_meta})
    summary = {
        "model_name": core.MODEL_NAME, "model_version": core.MODEL_VERSION,
        "feature_columns": list(core.FEATURE_COLUMNS), "diagnostic_version": DIAGNOSTIC_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(), "symbol": symbol,
        "timeframe": timeframe, "validation_mode": "STANDARD_WITH_LOCKBOX",
        "oos_samples": len(predictions), "folds": len(folds), **overall,
        "calibration_gap": calibration_gap, "train_mean_auc": train_auc,
        "validation_mean_auc": validation_auc,
        "diagnostic_oos_mean_auc": diagnostic_oos_auc,
        "fold_mean_auc": float(folds["oos_roc_auc"].mean()), "fold_auc_std": fold_std,
        "generalization_gap": train_auc - diagnostic_oos_auc,
        "overfit_status": _overfit_status(train_auc, diagnostic_oos_auc, fold_std),
        "folds_auc_above_0_5": float((folds["oos_roc_auc"] > 0.5).mean()),
        "best_trees_mean": float(folds["best_trees"].mean()),
        "best_trees_min": int(folds["best_trees"].min()),
        "best_trees_max": int(folds["best_trees"].max()),
        "auc_confidence_interval": block_bootstrap_auc(predictions["target_up"],
                                                        predictions["probability_up"]),
        "baselines": _baseline_summary(predictions),
        "noise_test": {"status": "COMPLETED" if noise_rows else "NOT_RUN",
                       "repeats_per_fold": NOISE_REPEATS if noise_rows else 0,
                       "top_20pct_rate": (float(pd.DataFrame(noise_rows)["noise_in_top_20pct"].mean())
                                          if noise_rows else None)},
        "lockbox": lockbox_status, "strategy": strategy,
        "reproducibility": reproducibility_info(raw, model_df, lockbox_meta),
    }
    core._atomic_write_csv(predictions, paths.validation_predictions)
    core._atomic_write_csv(folds, paths.validation_folds)
    core._atomic_write_csv(search_df, paths.validation_search)
    core._atomic_write_csv(reliability, paths.validation_reliability)
    core._atomic_write_csv(pd.DataFrame(permutation_rows), paths.validation_permutation)
    core._atomic_write_csv(pd.DataFrame(noise_rows), paths.validation_noise)
    core._atomic_write_json(summary, paths.validation_summary)
    return summary


def run_walk_forward(symbol: str, timeframe: str, refresh: bool = True,
                     run_noise_test: bool = False) -> dict[str, Any]:
    paths = core.get_paths(symbol, timeframe)
    lock = core.FileLock(str(paths.validation_summary) + ".lock", timeout=30 * 60)
    try:
        with lock:
            return _run_walk_forward_impl(symbol, timeframe, refresh, run_noise_test)
    except core.Timeout as exc:
        raise RuntimeError("동일 조합의 검증이 이미 실행 중입니다.") from exc


def evaluate_lockbox_once(symbol: str, timeframe: str, refresh: bool = False,
                          force: bool = False) -> dict[str, Any]:
    """예약된 최종 90일을 한 번만 평가하고 결과를 별도 저장한다."""
    paths = core.get_paths(symbol, timeframe)
    lock = core.FileLock(str(paths.lockbox_summary) + ".lock", timeout=30 * 60)
    try:
        with lock:
            if paths.lockbox_summary.exists() and not force:
                with paths.lockbox_summary.open(encoding="utf-8") as file:
                    existing = json.load(file)
                if (existing.get("model_version") == core.MODEL_VERSION and
                        existing.get("feature_columns") == list(core.FEATURE_COLUMNS)):
                    return existing
            raw, _, model_df = core.prepare_data(symbol, timeframe, refresh=refresh)
            development, lockbox, lockbox_meta = split_development_lockbox(model_df)
            artifact = core.fit_operational_artifact(
                development, development["timestamp"].iloc[-1], symbol, timeframe)
            raw_probability = artifact["model"].predict_proba(lockbox[core.FEATURE_COLUMNS])[:, 1]
            probability = core.apply_calibrator(artifact["calibration_method"],
                                                artifact["calibrator"], raw_probability)
            frame = lockbox[["timestamp", "close", "future_return", "target_up"]].copy()
            frame["threshold"] = float(artifact["threshold"])
            frame["raw_probability_up"] = raw_probability
            frame["probability_up"] = probability
            frame["probability_down"] = 1 - probability
            frame["prediction_up"] = (probability >= 0.5).astype(int)
            for column, values in _baseline_probabilities(development, lockbox).items():
                frame[column] = values
            metrics = _safe_metrics(frame["target_up"], frame["probability_up"])
            reliability, gap = calibration_reliability(frame["target_up"], frame["probability_up"])
            strategy = strategy_summary(frame, timeframe)
            result = {
                "status": "EVALUATED_ONCE", "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "model_name": core.MODEL_NAME, "model_version": core.MODEL_VERSION,
                "diagnostic_version": DIAGNOSTIC_VERSION,
                "feature_columns": list(core.FEATURE_COLUMNS), "symbol": symbol,
                "timeframe": timeframe, **lockbox_meta, **metrics,
                "calibration_gap": gap, "calibration_method": artifact["calibration_method"],
                "threshold": float(artifact["threshold"]), "xgb_config": artifact["xgb_config"],
                "baselines": _baseline_summary(frame), "strategy": strategy,
                "integrity_note": ("이 파일 생성 이후에는 동일 Lockbox로 모델·피처·임계값을 "
                                   "재선택하지 않아야 합니다. 과거 연구자가 이미 본 기간인지는 "
                                   "코드만으로 보증할 수 없습니다."),
                "reproducibility": reproducibility_info(raw, model_df, lockbox_meta),
            }
            core._atomic_write_csv(frame, paths.lockbox_predictions)
            core._atomic_write_csv(reliability,
                                   paths.validation_dir / "lockbox_probability_reliability.csv")
            core._atomic_write_json(result, paths.lockbox_summary)
            summary = load_validation_summary(symbol, timeframe)
            if summary is not None:
                summary["lockbox"] = result
                core._atomic_write_json(summary, paths.validation_summary)
            return result
    except core.Timeout as exc:
        raise RuntimeError("동일 조합의 Lockbox 평가가 이미 실행 중입니다.") from exc


def load_validation_summary(symbol: str, timeframe: str) -> dict | None:
    path = core.get_paths(symbol, timeframe).validation_summary
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as file:
            summary = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    if summary.get("model_version") != core.MODEL_VERSION:
        return None
    if summary.get("feature_columns") != list(core.FEATURE_COLUMNS):
        return None
    if summary.get("diagnostic_version") != DIAGNOSTIC_VERSION:
        return None
    return summary


def build_paper_closed_trades(log_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["trade_id", "direction", "entry_time", "exit_time", "entry_price",
               "exit_price", "holding_bars", "net_return", "result"]
    if log_df.empty:
        return pd.DataFrame(columns=columns)
    ordered = log_df.sort_values("target_candle_start").reset_index(drop=True)
    current, rows = None, []
    for _, row in ordered.iterrows():
        signal = int(row["signal"])
        timestamp = pd.Timestamp(row["target_candle_start"])
        price = float(row["reference_close"])
        if current is not None and signal != current["signal"]:
            direction = current["signal"]
            asset_return = direction * (price / current["entry_price"] - 1)
            entry_fee = core.LEVERAGE * core.CAPITAL_FRACTION * core.ONE_WAY_FEE_RATE
            exit_fee = (core.LEVERAGE * core.CAPITAL_FRACTION *
                        (price / current["entry_price"]) * core.ONE_WAY_FEE_RATE)
            net = core.LEVERAGE * core.CAPITAL_FRACTION * asset_return - entry_fee - exit_fee
            rows.append({"trade_id": len(rows) + 1,
                         "direction": "LONG" if direction == 1 else "SHORT",
                         "entry_time": current["entry_time"], "exit_time": timestamp,
                         "entry_price": current["entry_price"], "exit_price": price,
                         "holding_bars": current["holding_bars"], "net_return": net,
                         "result": "WIN" if net > 0 else "LOSS" if net < 0 else "BREAKEVEN"})
            current = None
        if signal != 0:
            if current is None:
                current = {"signal": signal, "entry_time": timestamp,
                           "entry_price": price, "holding_bars": 1}
            else:
                current["holding_bars"] += 1
    return pd.DataFrame(rows, columns=columns)


def paper_performance(symbol: str, timeframe: str) -> dict[str, Any]:
    paths = core.get_paths(symbol, timeframe)
    log_df = core.load_paper_log(paths)
    trades = build_paper_closed_trades(log_df)
    core._atomic_write_csv(trades, paths.closed_trades)
    values = trades["net_return"].to_numpy(dtype=float) if len(trades) else np.array([])
    wins, losses = values[values > 0], values[values < 0]
    result = {
        "prediction_opportunities": len(log_df),
        "settled_prediction_bars": int((log_df["status"] == "settled").sum()) if len(log_df) else 0,
        "closed_trades": len(trades), "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": float(len(wins) / len(values)) if len(values) else None,
        "average_win": float(wins.mean()) if len(wins) else None,
        "average_loss": float(losses.mean()) if len(losses) else None,
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else None,
        "expectancy": float(values.mean()) if len(values) else None,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else None,
        "max_consecutive_losses": _max_loss_streak(values), "ruin_risk": _ruin_risk(values),
    }
    core._atomic_write_json(result, paths.performance)
    return result
