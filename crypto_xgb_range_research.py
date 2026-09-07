"""Model A 피처로 다음 1봉 수익률의 조건부 분위수를 연구한다.

기존 ``crypto_xgb_core.py``와 웹사이트는 수정하지 않는다. 방향분류 모델과
별도로 Q10/Q50/Q90 수익률을 예측하여 예상 하단·중앙·상단 구간을 만든다.

검증 원칙
---------
* 종료된 캔들 및 기존 Model A 22개 피처만 사용한다.
* 모든 분할은 시간순이며 shuffle하지 않는다.
* 다음 1봉 타깃이므로 각 경계에서 1봉 purge한다.
* fit → tune → calibration-fit → calibration-select → OOS 순서를 지킨다.
* 하이퍼파라미터와 분위수 보정법은 Test Fold에서 선택하지 않는다.
* 마지막 Holdout은 개발구간 선택이 끝난 뒤 한 번만 평가한다.

실행 예시
---------
python3 crypto_xgb_range_research.py --symbol BTC --timeframe 4h
python3 crypto_xgb_range_research.py --symbol ETH --timeframe 1h --no-refresh
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error
from xgboost import XGBRegressor

import crypto_xgb_core as core


QUANTILES = np.array([0.10, 0.50, 0.90], dtype=float)
PURGE_BARS = 1
MIN_INNER_ROWS = 50
MIN_OUTER_TRAIN_ROWS = 400
MIN_TEST_ROWS = 30
TARGET_INTERVAL_COVERAGE = 0.80
FEATURE_SIGNATURE = "|".join(core.FEATURE_COLUMNS)


def research_paths(symbol: str, timeframe: str) -> dict[str, Path]:
    root = core.get_paths(symbol, timeframe, create=True).root / "range_research"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "design": root / "study_design.json",
        "oos_predictions": root / "development_oos_predictions.csv",
        "fold_metrics": root / "fold_metrics.csv",
        "parameter_search": root / "parameter_search.csv",
        "development_summary": root / "development_summary.json",
        "final_predictions": root / "final_holdout_predictions.csv",
        "final_summary": root / "final_holdout_summary.json",
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def make_range_data(featured: pd.DataFrame) -> pd.DataFrame:
    """기존 Model A의 다음 1봉 수익률과 22개 피처를 연구 데이터로 만든다."""
    required = [
        "timestamp", "close", *core.FEATURE_COLUMNS,
        "future_return", "target_up",
    ]
    result = featured.loc[:, required].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    result = (
        result.dropna(subset=["timestamp", *core.FEATURE_COLUMNS, "future_return"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    if len(result) < MIN_OUTER_TRAIN_ROWS + MIN_TEST_ROWS + PURGE_BARS:
        raise ValueError(
            f"학습 가능 데이터가 {len(result):,}개로 분위수 연구에 부족합니다."
        )
    return result


def split_selection_data(model_df: pd.DataFrame):
    """내부 선택구간을 시간순 네 부분으로 나누고 각 경계를 purge한다."""
    n = len(model_df)
    calibration_start = n - int(n * core.CALIBRATION_RATIO)
    tune_start = calibration_start - int(n * core.TUNE_RATIO)
    fit_df = model_df.iloc[: tune_start - PURGE_BARS].copy()
    tune_df = model_df.iloc[
        tune_start : calibration_start - PURGE_BARS
    ].copy()
    calibration_all = model_df.iloc[calibration_start:].copy()
    middle = len(calibration_all) // 2
    calibration_fit = calibration_all.iloc[: middle - PURGE_BARS].copy()
    calibration_select = calibration_all.iloc[middle:].copy()
    sizes = {
        "fit": len(fit_df),
        "tune": len(tune_df),
        "calibration_fit": len(calibration_fit),
        "calibration_select": len(calibration_select),
    }
    if min(sizes.values()) < MIN_INNER_ROWS:
        raise ValueError(f"분위수 모델 내부 학습 구간이 부족합니다: {sizes}")
    return fit_df, tune_df, calibration_fit, calibration_select


def build_fold_schedule(
    development_df: pd.DataFrame,
    timeframe: str,
) -> list[tuple[int, int, int, int]]:
    config = core.timeframe_config(timeframe)
    first_time = development_df["timestamp"].iloc[0]
    final_time = development_df["timestamp"].iloc[-1]
    test_start_time = first_time + pd.Timedelta(days=config.initial_train_days)
    schedule: list[tuple[int, int, int, int]] = []
    while test_start_time <= final_time:
        test_end_time = test_start_time + pd.Timedelta(days=config.test_days)
        test_start = int(
            development_df["timestamp"].searchsorted(test_start_time, side="left")
        )
        test_end = int(
            development_df["timestamp"].searchsorted(test_end_time, side="left")
        )
        train_end = test_start - PURGE_BARS
        if test_start >= len(development_df):
            break
        if train_end >= MIN_OUTER_TRAIN_ROWS and test_end - test_start >= MIN_TEST_ROWS:
            schedule.append((0, train_end, test_start, test_end))
        test_start_time += pd.Timedelta(days=config.step_days)
    return schedule


def _make_regressor(config, n_estimators: int, early_stopping: bool) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=QUANTILES,
        tree_method="hist",
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        learning_rate=config.learning_rate,
        n_estimators=int(n_estimators),
        reg_lambda=config.reg_lambda,
        reg_alpha=config.reg_alpha,
        early_stopping_rounds=(core.EARLY_STOPPING_ROUNDS if early_stopping else None),
        random_state=core.RANDOM_STATE,
        n_jobs=1,
    )


def _prediction_matrix(model: XGBRegressor, features: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model.predict(features), dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[1] != len(QUANTILES):
        raise ValueError(
            "XGBoost가 Q10/Q50/Q90 세 분위수를 반환하지 않았습니다. "
            "XGBoost 2.0 이상인지 확인하세요."
        )
    return values


def _ordered(prediction: np.ndarray) -> np.ndarray:
    """표시·평가 전에 행별 분위수 순서를 강제해 crossing을 제거한다."""
    return np.sort(np.asarray(prediction, dtype=float), axis=1)


def _pinball_by_quantile(y_true, prediction: np.ndarray) -> list[float]:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    return [
        float(mean_pinball_loss(y, pred[:, index], alpha=float(alpha)))
        for index, alpha in enumerate(QUANTILES)
    ]


def _range_metrics(y_true, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = _ordered(prediction)
    losses = _pinball_by_quantile(y, pred)
    lower, median, upper = pred[:, 0], pred[:, 1], pred[:, 2]
    covered = (y >= lower) & (y <= upper)
    alpha = 1.0 - TARGET_INTERVAL_COVERAGE
    interval_score = (
        (upper - lower)
        + (2.0 / alpha) * np.maximum(lower - y, 0.0)
        + (2.0 / alpha) * np.maximum(y - upper, 0.0)
    )
    return {
        "pinball_q10": losses[0],
        "pinball_q50": losses[1],
        "pinball_q90": losses[2],
        "mean_pinball": float(np.mean(losses)),
        "median_mae": float(mean_absolute_error(y, median)),
        "median_rmse": float(np.sqrt(mean_squared_error(y, median))),
        "interval_coverage": float(covered.mean()),
        "coverage_error": float(abs(covered.mean() - TARGET_INTERVAL_COVERAGE)),
        "mean_interval_width": float(np.mean(upper - lower)),
        "mean_interval_score": float(np.mean(interval_score)),
        "median_direction_accuracy": float(((median > 0) == (y > 0)).mean()),
    }


def _crossing_rate(prediction: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=float)
    return float(((pred[:, 0] > pred[:, 1]) | (pred[:, 1] > pred[:, 2])).mean())


def tune_regressor(
    fit_df: pd.DataFrame,
    tune_df: pd.DataFrame,
    fold: int,
) -> tuple[Any, int, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for config in core.XGB_CANDIDATES:
        model = _make_regressor(config, config.n_estimators, early_stopping=True)
        model.fit(
            fit_df[core.FEATURE_COLUMNS],
            fit_df["future_return"],
            eval_set=[(tune_df[core.FEATURE_COLUMNS], tune_df["future_return"])],
            verbose=False,
        )
        raw = _prediction_matrix(model, tune_df[core.FEATURE_COLUMNS])
        metrics = _range_metrics(tune_df["future_return"], raw)
        best_iteration = getattr(model, "best_iteration", None)
        best_trees = int(best_iteration) + 1 if best_iteration is not None else config.n_estimators
        rows.append(
            {
                "fold": fold,
                "config": config.name,
                "best_trees": best_trees,
                "raw_crossing_rate": _crossing_rate(raw),
                **metrics,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["mean_pinball", "coverage_error", "median_mae"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    winner = table.iloc[0]
    config = next(item for item in core.XGB_CANDIDATES if item.name == winner["config"])
    return config, int(winner["best_trees"]), table


def fit_quantile_offsets(y_true, raw_prediction: np.ndarray) -> np.ndarray:
    """calibration-fit 구간의 잔차 분위수로 Q10/Q50/Q90 위치를 보정한다."""
    y = np.asarray(y_true, dtype=float)
    raw = _ordered(raw_prediction)
    return np.asarray(
        [
            np.quantile(y - raw[:, index], float(alpha))
            for index, alpha in enumerate(QUANTILES)
        ],
        dtype=float,
    )


def apply_quantile_offsets(raw_prediction: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return _ordered(np.asarray(raw_prediction, dtype=float) + np.asarray(offsets))


def select_range_calibration(
    y_select,
    raw_select: np.ndarray,
    offsets: np.ndarray,
) -> tuple[str, pd.DataFrame]:
    candidates = {
        "raw": _ordered(raw_select),
        "offset": apply_quantile_offsets(raw_select, offsets),
    }
    rows = []
    for name, prediction in candidates.items():
        rows.append({"calibration": name, **_range_metrics(y_select, prediction)})
    table = pd.DataFrame(rows).sort_values(
        ["mean_pinball", "coverage_error", "median_mae"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return str(table.iloc[0]["calibration"]), table


def run_fold(
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    fold: int,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    fit_df, tune_df, calibration_fit, calibration_select = split_selection_data(
        outer_train
    )
    config, best_trees, search = tune_regressor(fit_df, tune_df, fold)
    pre_calibration = pd.concat([fit_df, tune_df], ignore_index=True)
    model = _make_regressor(config, best_trees, early_stopping=False)
    model.fit(
        pre_calibration[core.FEATURE_COLUMNS],
        pre_calibration["future_return"],
        verbose=False,
    )

    raw_fit = _prediction_matrix(model, calibration_fit[core.FEATURE_COLUMNS])
    offsets = fit_quantile_offsets(calibration_fit["future_return"], raw_fit)
    raw_select = _prediction_matrix(model, calibration_select[core.FEATURE_COLUMNS])
    calibration, calibration_table = select_range_calibration(
        calibration_select["future_return"], raw_select, offsets
    )

    raw_test = _prediction_matrix(model, outer_test[core.FEATURE_COLUMNS])
    prediction = (
        apply_quantile_offsets(raw_test, offsets)
        if calibration == "offset"
        else _ordered(raw_test)
    )
    metrics = _range_metrics(outer_test["future_return"], prediction)

    baseline_values = np.quantile(
        np.asarray(outer_train["future_return"], dtype=float), QUANTILES
    )
    baseline = np.tile(baseline_values, (len(outer_test), 1))
    baseline_metrics = _range_metrics(outer_test["future_return"], baseline)

    predictions = outer_test[
        ["timestamp", "close", "future_return", "target_up"]
    ].copy()
    predictions["fold"] = fold
    predictions["q10_return"] = prediction[:, 0]
    predictions["q50_return"] = prediction[:, 1]
    predictions["q90_return"] = prediction[:, 2]
    predictions["q10_price"] = predictions["close"] * (1 + prediction[:, 0])
    predictions["q50_price"] = predictions["close"] * (1 + prediction[:, 1])
    predictions["q90_price"] = predictions["close"] * (1 + prediction[:, 2])
    predictions["baseline_q10_return"] = baseline[:, 0]
    predictions["baseline_q50_return"] = baseline[:, 1]
    predictions["baseline_q90_return"] = baseline[:, 2]
    predictions["config"] = config.name
    predictions["best_trees"] = best_trees
    predictions["range_calibration"] = calibration

    summary = {
        "fold": fold,
        "train_start": str(outer_train["timestamp"].iloc[0]),
        "train_end": str(outer_train["timestamp"].iloc[-1]),
        "test_start": str(outer_test["timestamp"].iloc[0]),
        "test_end": str(outer_test["timestamp"].iloc[-1]),
        "samples": len(outer_test),
        "config": config.name,
        "best_trees": best_trees,
        "range_calibration": calibration,
        "offset_q10": float(offsets[0]),
        "offset_q50": float(offsets[1]),
        "offset_q90": float(offsets[2]),
        "raw_test_crossing_rate": _crossing_rate(raw_test),
        "baseline_mean_pinball": baseline_metrics["mean_pinball"],
        **metrics,
    }
    search = search.copy()
    search["selected_calibration"] = calibration
    search["calibration_comparison"] = json.dumps(
        calibration_table.to_dict(orient="records"), ensure_ascii=False
    )
    return predictions, summary, search


def summarize_predictions(
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
) -> dict[str, Any]:
    prediction = predictions[["q10_return", "q50_return", "q90_return"]].to_numpy()
    baseline = predictions[
        ["baseline_q10_return", "baseline_q50_return", "baseline_q90_return"]
    ].to_numpy()
    metrics = _range_metrics(predictions["future_return"], prediction)
    baseline_metrics = _range_metrics(predictions["future_return"], baseline)
    skill = (
        1.0 - metrics["mean_pinball"] / baseline_metrics["mean_pinball"]
        if baseline_metrics["mean_pinball"] > 0
        else 0.0
    )
    return {
        "oos_samples": int(len(predictions)),
        "folds": int(len(folds)),
        **metrics,
        "baseline_mean_pinball": baseline_metrics["mean_pinball"],
        "pinball_skill_score": float(skill),
        "fold_mean_pinball": float(folds["mean_pinball"].mean()),
        "fold_pinball_std": (
            float(folds["mean_pinball"].std(ddof=1)) if len(folds) > 1 else 0.0
        ),
        "fold_mean_coverage": float(folds["interval_coverage"].mean()),
        "folds_positive_skill_ratio": float(
            (folds["mean_pinball"] < folds["baseline_mean_pinball"]).mean()
        ),
        "raw_crossing_rate": float(
            np.average(folds["raw_test_crossing_rate"], weights=folds["samples"])
        ),
    }


def _load_or_create_design(
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    if path.exists():
        with path.open(encoding="utf-8") as file:
            design = json.load(file)
        if design.get("symbol") != symbol or design.get("timeframe") != timeframe:
            raise ValueError("기존 구간 연구의 코인·시간봉이 현재 선택과 다릅니다.")
        if design.get("feature_signature") != FEATURE_SIGNATURE:
            raise ValueError("기존 연구와 현재 Model A의 피처 구성이 다릅니다.")
        if design.get("quantiles") != QUANTILES.tolist():
            raise ValueError("기존 연구와 현재 분위수 후보가 다릅니다.")
        return design

    config = core.timeframe_config(timeframe)
    holdout_end = pd.Timestamp(raw_df["timestamp"].max())
    holdout_start = holdout_end - pd.Timedelta(days=config.test_days)
    design = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "target": "next_1_bar_return",
        "purge_bars": PURGE_BARS,
        "quantiles": QUANTILES.tolist(),
        "target_interval_coverage": TARGET_INTERVAL_COVERAGE,
        "feature_signature": FEATURE_SIGNATURE,
        "xgboost_version": xgboost.__version__,
        "holdout_start": str(holdout_start),
        "holdout_end": str(holdout_end),
        "warning": "Holdout 결과를 본 뒤 모델 규칙을 바꾸면 더 이상 최종 Test가 아니다.",
    }
    _write_json(design, path)
    return design


def split_development_holdout(
    model_df: pd.DataFrame,
    design: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_start = pd.to_datetime(design["holdout_start"])
    holdout_end = pd.to_datetime(design["holdout_end"])
    start_index = int(model_df["timestamp"].searchsorted(holdout_start, side="left"))
    development = model_df.iloc[: start_index - PURGE_BARS].copy()
    timeframe_delta = core.timeframe_config(str(design["timeframe"])).delta
    label_end = model_df["timestamp"] + timeframe_delta
    holdout = model_df.loc[
        (model_df["timestamp"] >= holdout_start)
        & (model_df["timestamp"] <= holdout_end)
        & (label_end <= holdout_end)
    ].copy()
    if len(development) < MIN_OUTER_TRAIN_ROWS:
        raise ValueError("고정 Holdout 이전 개발 데이터가 부족합니다.")
    if len(holdout) < MIN_TEST_ROWS:
        raise ValueError("고정 최종 Holdout 표본이 부족합니다.")
    return development, holdout


def run_range_research(
    symbol: str,
    timeframe: str,
    refresh: bool = True,
    evaluate_final: bool = True,
    force_final: bool = False,
) -> dict[str, Any]:
    core.validate_selection(symbol, timeframe)
    major_version = int(str(xgboost.__version__).split(".")[0])
    if major_version < 2:
        raise RuntimeError("분위수 회귀에는 XGBoost 2.0 이상이 필요합니다.")

    core.report(f"\n[{symbol} {timeframe} 다음 1봉 예상 구간 연구]")
    core.report("분위수: Q10 / Q50 / Q90")
    raw = core.load_raw_data(symbol, timeframe, refresh=refresh)
    featured = core.add_features(raw)
    model_df = make_range_data(featured)
    paths = research_paths(symbol, timeframe)
    design = _load_or_create_design(symbol, timeframe, raw, paths["design"])
    development, holdout = split_development_holdout(model_df, design)
    schedule = build_fold_schedule(development, timeframe)
    if not schedule:
        raise ValueError("개발구간 Walk-Forward Fold를 만들 수 없습니다.")
    core.report(f"고정 Holdout: {design['holdout_start']} ~ {design['holdout_end']}")
    core.report(f"개발 Fold: {len(schedule)}개")

    prediction_frames = []
    fold_rows = []
    search_frames = []
    for fold, (train_start, train_end, test_start, test_end) in enumerate(
        schedule, start=1
    ):
        outer_train = development.iloc[train_start:train_end].copy()
        outer_test = development.iloc[test_start:test_end].copy()
        predictions, fold_summary, search = run_fold(outer_train, outer_test, fold)
        prediction_frames.append(predictions)
        fold_rows.append(fold_summary)
        search_frames.append(search)
        skill = 1 - fold_summary["mean_pinball"] / fold_summary["baseline_mean_pinball"]
        core.report(
            f"Fold {fold:02d}/{len(schedule):02d} | "
            f"Pinball={fold_summary['mean_pinball']:.6f} | "
            f"Skill={skill:+.2%} | "
            f"Coverage={fold_summary['interval_coverage']:.1%} | "
            f"{fold_summary['config']} | {fold_summary['range_calibration']}"
        )

    oos_predictions = pd.concat(prediction_frames, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    searches = pd.concat(search_frames, ignore_index=True)
    development_summary = summarize_predictions(oos_predictions, folds)
    development_summary.update(
        {
            "evaluation": "DEVELOPMENT_EXPANDING_WALK_FORWARD",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "quantiles": QUANTILES.tolist(),
        }
    )
    _write_csv(oos_predictions, paths["oos_predictions"])
    _write_csv(folds, paths["fold_metrics"])
    _write_csv(searches, paths["parameter_search"])
    _write_json(development_summary, paths["development_summary"])

    core.report("\n[개발구간 OOS 예상 구간 품질]")
    core.report(f"평균 Pinball Loss : {development_summary['mean_pinball']:.6f}")
    core.report(f"Pinball Skill     : {development_summary['pinball_skill_score']:+.2%}")
    core.report(f"Q10~Q90 포함률    : {development_summary['interval_coverage']:.2%} (목표 80%)")
    core.report(f"포함률 오차       : {development_summary['coverage_error']:.2%}p")
    core.report(f"중앙값 MAE        : {development_summary['median_mae']:.4%}")
    core.report(f"평균 예상 구간 폭 : {development_summary['mean_interval_width']:.4%}")
    core.report(f"양의 Skill Fold   : {development_summary['folds_positive_skill_ratio']:.2%}")
    core.report(f"원시 분위수 교차율: {development_summary['raw_crossing_rate']:.2%}")

    final_summary: dict[str, Any] | None = None
    if evaluate_final:
        if paths["final_summary"].exists() and not force_final:
            with paths["final_summary"].open(encoding="utf-8") as file:
                final_summary = json.load(file)
            core.report("[최종 Holdout] 기존 결과를 보존하고 재평가하지 않았습니다.")
        else:
            final_predictions, fold_summary, final_search = run_fold(
                development, holdout, fold=0
            )
            final_summary = summarize_predictions(
                final_predictions, pd.DataFrame([fold_summary])
            )
            final_summary.update(
                {
                    "evaluation": "FROZEN_FINAL_HOLDOUT",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "holdout_start": design["holdout_start"],
                    "holdout_end": design["holdout_end"],
                    "selected_config": fold_summary["config"],
                    "selected_trees": fold_summary["best_trees"],
                    "selected_calibration": fold_summary["range_calibration"],
                    "parameter_search": final_search.drop(
                        columns=["calibration_comparison"], errors="ignore"
                    ).to_dict(orient="records"),
                }
            )
            _write_csv(final_predictions, paths["final_predictions"])
            _write_json(final_summary, paths["final_summary"])
            core.report("\n[고정 최종 Holdout — 한 번만 해석]")
            core.report(f"평균 Pinball Loss : {final_summary['mean_pinball']:.6f}")
            core.report(f"Pinball Skill     : {final_summary['pinball_skill_score']:+.2%}")
            core.report(f"Q10~Q90 포함률    : {final_summary['interval_coverage']:.2%}")
            core.report(f"중앙값 MAE        : {final_summary['median_mae']:.4%}")

    core.report(f"\n[저장 위치] {paths['root']}")
    return {
        "design": design,
        "development_summary": development_summary,
        "final_summary": final_summary,
        "paths": paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XGBoost 다음 1봉 분위수 구간 연구",
        allow_abbrev=False,
    )
    parser.add_argument("--symbol", help="BTC, ETH, SOL 또는 USD-M 토큰명")
    parser.add_argument("--timeframe", choices=list(core.TIMEFRAME_CONFIGS))
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument(
        "--force-final",
        action="store_true",
        help="기존 최종 Holdout 결과를 덮어쓴다. 연구상 권장하지 않는다.",
    )
    args, _ = parser.parse_known_args()
    return args


def _interactive_selection(args: argparse.Namespace) -> tuple[str, str]:
    token = (args.symbol or input("토큰명 (예: BTC): ")).strip().upper()
    timeframe = (args.timeframe or input("시간봉 (15m/1h/4h/1d): ")).strip()
    symbol = core.symbol_from_token(token)
    core.validate_selection(symbol, timeframe)
    return symbol, timeframe


def main() -> None:
    args = parse_args()
    try:
        symbol, timeframe = _interactive_selection(args)
        run_range_research(
            symbol=symbol,
            timeframe=timeframe,
            refresh=not args.no_refresh,
            evaluate_final=not args.skip_final,
            force_final=args.force_final,
        )
    except (EOFError, KeyboardInterrupt):
        core.report("\n실행이 중단되었습니다.")
    except Exception as exc:
        core.report(f"\n[실행 실패] {exc}")


if __name__ == "__main__":
    main()
