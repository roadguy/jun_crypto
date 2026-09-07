"""다음 1봉 Q10/Q50/Q90 운영 예측.

방향 분류 Model A와 독립적으로 동작하며 실제 주문을 전송하지 않는다.
연구 파일의 튜닝·분위수 보정 함수를 재사용하고, 고정 Holdout 결과는
모델 선택에 다시 사용하지 않는다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import crypto_xgb_core as core
import crypto_xgb_range_research as research


RANGE_MODEL_VERSION = "xgb_model_a_q10_q50_q90_v1"


def live_paths(symbol: str, timeframe: str) -> dict[str, Path]:
    root = core.get_paths(symbol, timeframe, create=True).root / "range_live"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "model": root / "range_model.joblib",
        "metadata": root / "range_model_metadata.json",
    }


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)


def _metadata(artifact: dict[str, Any]) -> dict[str, Any]:
    hidden = {"model", "offsets"}
    payload = {key: value for key, value in artifact.items() if key not in hidden}
    payload["offsets"] = np.asarray(artifact["offsets"], dtype=float).tolist()
    payload["actual_orders_enabled"] = False
    return payload


def save_artifact(artifact: dict[str, Any], symbol: str, timeframe: str) -> None:
    paths = live_paths(symbol, timeframe)
    temporary = paths["model"].with_suffix(".joblib.tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(paths["model"])
    _atomic_json(_metadata(artifact), paths["metadata"])


def load_artifact(symbol: str, timeframe: str) -> dict[str, Any] | None:
    path = live_paths(symbol, timeframe)["model"]
    if not path.exists():
        return None
    try:
        value = joblib.load(path)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def artifact_is_current(
    artifact: dict[str, Any] | None,
    symbol: str,
    timeframe: str,
) -> bool:
    if artifact is None:
        return False
    if artifact.get("model_version") != RANGE_MODEL_VERSION:
        return False
    if artifact.get("feature_columns") != core.FEATURE_COLUMNS:
        return False
    if artifact.get("symbol") != symbol or artifact.get("timeframe") != timeframe:
        return False
    trained_at = pd.to_datetime(artifact.get("trained_at"), utc=True, errors="coerce")
    if pd.isna(trained_at):
        return False
    age = pd.Timestamp.now(tz="UTC") - trained_at
    return age < pd.Timedelta(days=core.timeframe_config(timeframe).retrain_days)


def fit_live_artifact(
    model_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    latest_source_timestamp: Any,
) -> dict[str, Any]:
    """과거 라벨 구간에서만 튜닝·보정법을 선택해 운영 모델을 만든다."""
    fit_df, tune_df, calibration_fit, calibration_select = (
        research.split_selection_data(model_df)
    )
    config, best_trees, parameter_search = research.tune_regressor(
        fit_df, tune_df, fold=0
    )

    pre_calibration = pd.concat([fit_df, tune_df], ignore_index=True)
    model = research._make_regressor(config, best_trees, early_stopping=False)
    model.fit(
        pre_calibration[core.FEATURE_COLUMNS],
        pre_calibration["future_return"],
        verbose=False,
    )

    raw_fit = research._prediction_matrix(
        model, calibration_fit[core.FEATURE_COLUMNS]
    )
    offsets = research.fit_quantile_offsets(
        calibration_fit["future_return"], raw_fit
    )
    raw_select = research._prediction_matrix(
        model, calibration_select[core.FEATURE_COLUMNS]
    )
    calibration, calibration_search = research.select_range_calibration(
        calibration_select["future_return"], raw_select, offsets
    )

    return {
        "model_name": "Model A Quantile Range",
        "model_version": RANGE_MODEL_VERSION,
        "model": model,
        "offsets": np.asarray(offsets, dtype=float),
        "calibration_method": calibration,
        "quantiles": research.QUANTILES.tolist(),
        "xgb_config": config.name,
        "best_trees": int(best_trees),
        "feature_columns": list(core.FEATURE_COLUMNS),
        "symbol": symbol,
        "timeframe": timeframe,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "latest_source_timestamp": str(latest_source_timestamp),
        "available_labeled_end": str(model_df["timestamp"].iloc[-1]),
        "available_labeled_samples": int(len(model_df)),
        "parameter_search": parameter_search.to_dict(orient="records"),
        "calibration_search": calibration_search.to_dict(orient="records"),
    }


def get_or_train_artifact(
    model_df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    latest_source_timestamp: Any,
    force_retrain: bool = False,
) -> tuple[dict[str, Any], bool]:
    artifact = load_artifact(symbol, timeframe)
    reused = not force_retrain and artifact_is_current(
        artifact, symbol, timeframe
    )
    if not reused:
        artifact = fit_live_artifact(
            model_df, symbol, timeframe, latest_source_timestamp
        )
        save_artifact(artifact, symbol, timeframe)
    return artifact, reused


def load_quality_summary(symbol: str, timeframe: str) -> dict[str, Any] | None:
    """최종 Holdout을 우선 표시하고, 없으면 개발 OOS 요약을 사용한다."""
    paths = research.research_paths(symbol, timeframe)
    candidates = [
        (paths["final_summary"], "FROZEN_FINAL_HOLDOUT"),
        (paths["development_summary"], "DEVELOPMENT_EXPANDING_WALK_FORWARD"),
    ]
    for path, fallback_name in candidates:
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8") as file:
                payload = json.load(file)
            if isinstance(payload, dict):
                payload.setdefault("evaluation", fallback_name)
                return payload
        except (OSError, json.JSONDecodeError):
            continue
    return None


def run_latest_range_prediction(
    symbol: str,
    timeframe: str,
    refresh: bool = False,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """최신 종료봉으로 다음 1봉의 예상 하단·중앙·상단을 반환한다."""
    core.validate_selection(symbol, timeframe)
    raw = core.load_raw_data(symbol, timeframe, refresh=refresh)
    featured = core.add_features(raw)
    model_df = research.make_range_data(featured)
    latest = (
        featured.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=core.FEATURE_COLUMNS)
        .sort_values("timestamp")
        .tail(1)
    )
    if latest.empty:
        raise ValueError("분위수 최신 예측에 필요한 피처가 없습니다.")

    latest_timestamp = pd.Timestamp(latest["timestamp"].iloc[0])
    artifact, reused = get_or_train_artifact(
        model_df=model_df,
        symbol=symbol,
        timeframe=timeframe,
        latest_source_timestamp=latest_timestamp,
        force_retrain=force_retrain,
    )
    raw_prediction = research._prediction_matrix(
        artifact["model"], latest[core.FEATURE_COLUMNS]
    )
    prediction = (
        research.apply_quantile_offsets(raw_prediction, artifact["offsets"])
        if artifact["calibration_method"] == "offset"
        else research._ordered(raw_prediction)
    )[0]

    reference_close = float(latest["close"].iloc[0])
    config = core.timeframe_config(timeframe)
    source_end = latest_timestamp + config.delta
    target_start = source_end
    target_end = target_start + config.delta
    prices = reference_close * (1.0 + prediction)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "source_candle_start": latest_timestamp,
        "source_candle_end": source_end,
        "target_candle_start": target_start,
        "target_candle_end": target_end,
        "reference_close": reference_close,
        "q10_return": float(prediction[0]),
        "q50_return": float(prediction[1]),
        "q90_return": float(prediction[2]),
        "q10_price": float(prices[0]),
        "q50_price": float(prices[1]),
        "q90_price": float(prices[2]),
        "calibration_method": artifact["calibration_method"],
        "trained_at": artifact["trained_at"],
        "model_reused": reused,
        "quality_summary": load_quality_summary(symbol, timeframe),
        "paths": live_paths(symbol, timeframe),
    }

