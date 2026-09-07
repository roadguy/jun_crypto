"""FastAPI server for the HTML cryptocurrency research dashboard.

The browser only renders the interface.  Every data/model operation continues
to use the existing, tested Model A modules.  Long operations run in a single
background worker and are polled by the browser, which prevents a web request
from appearing frozen while XGBoost is training.
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import crypto_xgb_core as core
import crypto_xgb_range_live as range_live
import crypto_xgb_validation as validation


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"

app = FastAPI(
    title="Signal Ledger API",
    description="Model A 연구용 암호화폐 방향 예측 API",
    version="1.0.0",
)


class JobRequest(BaseModel):
    action: Literal["predict", "refresh", "validate", "lockbox"]
    token: str = Field(default="BTC", min_length=2, max_length=30)
    timeframe: Literal["15m", "1h", "4h", "1d"] = "4h"
    noise_test: bool = False
    confirm_lockbox: bool = False


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
# RAM 512 MB 환경을 고려해 무거운 모델 작업은 한 번에 하나만 실행한다.
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-job")
MAX_QUEUED_JOBS = 3


FEATURE_GROUPS = [
    {
        "title": "가격 움직임",
        "plain": "최근 가격이 얼마나 오르거나 내렸는지 봅니다.",
        "features": ["return_1", "return_3", "return_12", "rsi_14"],
    },
    {
        "title": "캔들 모양",
        "plain": "몸통과 꼬리 모양, 종가 위치, 같은 방향의 연속 횟수를 봅니다.",
        "features": [
            "range_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "close_position", "streak_count",
        ],
    },
    {
        "title": "거래량",
        "plain": "평소보다 거래가 몰렸는지와 상승 캔들 거래량을 봅니다.",
        "features": ["volume_change", "volume_zscore_20", "up_volume_ratio_10"],
    },
    {
        "title": "시장 흔들림",
        "plain": "가격이 조용한지 크게 흔들리는지, 밴드 안 어디에 있는지 봅니다.",
        "features": ["volatility_20", "atr_14_ratio", "bb_pct", "parkinson_vol_20"],
    },
    {
        "title": "큰 흐름",
        "plain": "현재 가격이 EMA 50과 EMA 200 위인지 아래인지 봅니다.",
        "features": ["close_ema_50_ratio", "close_ema_200_ratio"],
    },
    {
        "title": "시간",
        "plain": "하루 중 시간과 요일의 반복 패턴을 숫자로 바꿔 봅니다.",
        "features": ["hour_sin", "hour_cos", "day_sin", "day_cos"],
    },
]


def _selection(token: str, timeframe: str) -> tuple[str, str]:
    clean_token = str(token).strip().upper()
    try:
        symbol = core.symbol_from_token(clean_token)
        core.validate_selection(symbol, timeframe)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return clean_token, symbol


def _clean(value: Any) -> Any:
    """Convert pandas/numpy/path objects into strict JSON-compatible values."""
    if is_dataclass(value):
        return _clean(asdict(value))
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(item) for item in value]
    return value


def _friendly_error(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    lowered = message.lower()
    if "451" in message or "403" in message or "restricted location" in lowered:
        return message
    if "ssl" in lowered or "certificate" in lowered:
        return "데이터 서버의 보안 인증서를 확인하지 못했습니다. 잠시 후 다시 시도하세요."
    if "insufficient" in lowered or "부족" in message:
        return message
    if "timeout" in lowered:
        return "작업 시간이 길어졌습니다. 잠시 후 다시 확인하세요."
    return message


def _read_snapshot(symbol: str, timeframe: str) -> dict[str, Any]:
    paths = core.get_paths(symbol, timeframe)
    raw = None
    if paths.raw_ohlcv.exists():
        raw = core.load_raw_data(symbol, timeframe, refresh=False)
    summary = validation.load_validation_summary(symbol, timeframe)
    if summary is None:
        # 포트폴리오 배포 직후에도 이미 완료한 OOS 결과를 보여준다. 이 파일은
        # 화면 표시용 복사본일 뿐 새 모델의 선택이나 학습에는 사용하지 않는다.
        seed_summary = (
            ROOT / "portfolio_results" / core.symbol_slug(symbol)
            / timeframe / "summary.json"
        )
        try:
            with seed_summary.open(encoding="utf-8") as file:
                candidate = json.load(file)
            if (
                candidate.get("model_version") == core.MODEL_VERSION
                and candidate.get("feature_columns") == list(core.FEATURE_COLUMNS)
            ):
                summary = candidate
        except (OSError, json.JSONDecodeError):
            pass
    last_end = (pd.Timestamp(raw["timestamp"].iloc[-1]) + core.timeframe_config(timeframe).delta) if raw is not None and len(raw) else None
    stale = last_end is None or pd.Timestamp.now(tz="UTC") >= last_end + core.timeframe_config(timeframe).delta
    return {
        "data_source": "Binance USD-M perpetual",
        "is_stale": stale,
        "last_closed_at": last_end,
        "has_data": raw is not None and not raw.empty,
        "candle_count": int(len(raw)) if raw is not None else 0,
        "first_candle": raw["timestamp"].iloc[0] if raw is not None and len(raw) else None,
        "last_candle": raw["timestamp"].iloc[-1] if raw is not None and len(raw) else None,
        "validation_summary": summary,
    }


def _execute_job(job_id: str, request: JobRequest) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(status="running", message="작업을 시작했습니다.")

    def progress(message):
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["message"] = message
            job["log"] = (job.get("log", "") + message + "\n")[-20000:]
    callback_token = core.PROGRESS_CALLBACK.set(progress)
    try:
        token, symbol = _selection(request.token, request.timeframe)
        payload: dict[str, Any]
        if request.action == "predict":
            result = core.run_latest_prediction(
                symbol, request.timeframe, refresh=True, force_retrain=False
            )
            progress("방향 예측 완료. 예상 가격 구간을 계산합니다.")
            warnings = []
            try:
                range_result = range_live.run_latest_range_prediction(
                    symbol, request.timeframe, refresh=False, force_retrain=False
                )
            except Exception as exc:
                range_result = None
                warnings.append("가격 구간 계산 실패: " + _friendly_error(exc))
            payload_snapshot = _read_snapshot(symbol, request.timeframe)
            if payload_snapshot["is_stale"]:
                warnings.append("최신 종료 캔들이 누락되어 있습니다. 표시된 예측 기준 시각을 확인하세요.")
            payload = {
                "prediction": result,
                "range_prediction": range_result,
                "warnings": warnings,
                **payload_snapshot,
            }
            message = "최신 예측이 준비되었습니다."
        elif request.action == "refresh":
            raw = core.load_raw_data(symbol, request.timeframe, refresh=True)
            payload = {
                **_read_snapshot(symbol, request.timeframe),
                "candle_count": int(len(raw)),
            }
            message = f"종료된 캔들 {len(raw):,}개를 확인했습니다."
        elif request.action == "validate":
            summary = validation.run_walk_forward(
                symbol,
                request.timeframe,
                refresh=False,
                run_noise_test=request.noise_test,
            )
            payload = {
                **_read_snapshot(symbol, request.timeframe),
                "validation_summary": summary,
            }
            message = "개발구간 Walk-Forward 검증이 완료되었습니다."
        else:
            if not request.confirm_lockbox:
                raise ValueError("최종 미공개 데이터 검증 확인란을 먼저 체크하세요.")
            lockbox = validation.evaluate_lockbox_once(
                symbol, request.timeframe, refresh=False, force=False
            )
            payload = {
                **_read_snapshot(symbol, request.timeframe),
                "lockbox_result": lockbox,
            }
            message = "최종 미공개 데이터 검증이 완료되었습니다."

        payload.update(token=token, symbol=symbol, timeframe=request.timeframe)
        with JOBS_LOCK:
            JOBS[job_id].update(
                status="complete",
                message=message,
                result=_clean(payload),
            )
    except Exception as exc:  # The API must return the model's useful failure text.
        with JOBS_LOCK:
            JOBS[job_id].update(
                status="failed",
                message=_friendly_error(exc),
                error_type=type(exc).__name__,
            )

    finally:
        core.PROGRESS_CALLBACK.reset(callback_token)

def _chart_payload(raw: pd.DataFrame, limit: int = 2000) -> dict[str, Any]:
    chart = raw[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    chart["timestamp"] = pd.to_datetime(chart["timestamp"], utc=True, errors="coerce")
    chart = chart.dropna().sort_values("timestamp")
    close = pd.to_numeric(chart["close"], errors="coerce")
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std(ddof=0)
    upper, lower = middle + 2 * deviation, middle - 2 * deviation

    candles, volume, indicators = [], [], {
        "ema50": [], "ema200": [], "bb_upper": [], "bb_middle": [], "bb_lower": []
    }
    # Calculate indicators with full history, serialize only the most recent 2,000.
    for index, row in chart.tail(limit).iterrows():
        unix_time = int(row["timestamp"].timestamp())
        open_price, close_price = float(row["open"]), float(row["close"])
        candles.append({
            "time": unix_time,
            "open": open_price,
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": close_price,
        })
        volume.append({
            "time": unix_time,
            "value": float(row["volume"]),
            "color": "rgba(52, 211, 153, .42)" if close_price >= open_price else "rgba(251, 113, 133, .42)",
        })
        for name, series in (
            ("ema50", ema50), ("ema200", ema200),
            ("bb_upper", upper), ("bb_middle", middle), ("bb_lower", lower),
        ):
            value = series.loc[index]
            if pd.notna(value):
                indicators[name].append({"time": unix_time, "value": float(value)})
    return {
        "candles": candles,
        "volume": volume,
        "indicators": indicators,
        "count": len(candles),
        "total_count": len(chart),
        "last_price": candles[-1]["close"] if candles else None,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": core.MODEL_NAME,
        "model_version": core.MODEL_VERSION,
        "orders_enabled": False,
    }


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "timeframes": [
            {"value": "15m", "label": "15분"},
            {"value": "1h", "label": "1시간"},
            {"value": "4h", "label": "4시간"},
            {"value": "1d", "label": "1일"},
        ],
        "feature_count": len(core.FEATURE_COLUMNS),
        "feature_groups": FEATURE_GROUPS,
        "leverage": core.LEVERAGE,
        "fee_rate": core.ONE_WAY_FEE_RATE,
    }


@app.get("/api/snapshot")
def snapshot(token: str = "BTC", timeframe: str = "4h") -> dict[str, Any]:
    clean_token, symbol = _selection(token, timeframe)
    return _clean({
        "token": clean_token,
        "symbol": symbol,
        "timeframe": timeframe,
        **_read_snapshot(symbol, timeframe),
    })


@app.get("/api/chart")
def chart(token: str = "BTC", timeframe: str = "4h", limit: int = Query(default=2000, ge=200, le=100000)) -> dict[str, Any]:
    _, symbol = _selection(token, timeframe)
    paths = core.get_paths(symbol, timeframe)
    if not paths.raw_ohlcv.exists():
        raise HTTPException(status_code=404, detail="저장된 캔들이 없습니다. 최신 예측 또는 데이터 새로고침을 먼저 실행하세요.")
    raw = core.load_raw_data(symbol, timeframe, refresh=False)
    return _clean(_chart_payload(raw, limit))


@app.post("/api/jobs", status_code=202)
def start_job(request: JobRequest) -> dict[str, str]:
    _, symbol = _selection(request.token, request.timeframe)
    if request.action == "lockbox" and not request.confirm_lockbox:
        raise HTTPException(status_code=422, detail="최종 미공개 데이터 검증 확인란을 먼저 체크하세요.")
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        active = [
            job for job in JOBS.values()
            if job.get("status") in {"queued", "running"}
        ]
        if len(active) >= MAX_QUEUED_JOBS:
            raise HTTPException(
                status_code=429,
                detail="이미 여러 계산이 대기 중입니다. 앞선 작업이 끝난 뒤 다시 시도하세요.",
            )
        duplicate_key = f"{request.action}|{symbol}|{request.timeframe}"
        if any(job.get("key") == duplicate_key for job in active):
            raise HTTPException(
                status_code=409,
                detail="같은 코인·시간봉 작업이 이미 실행 또는 대기 중입니다.",
            )
        # 장시간 공개 운영 시 완료 작업이 메모리에 끝없이 쌓이지 않도록 제한한다.
        completed_ids = [
            key for key, job in JOBS.items()
            if job.get("status") in {"complete", "failed"}
        ]
        for old_id in completed_ids[:-50]:
            JOBS.pop(old_id, None)
        JOBS[job_id] = {
            "id": job_id,
            "key": duplicate_key,
            "status": "queued",
            "action": request.action,
            "message": "앞선 작업이 있으면 끝난 뒤 자동으로 시작합니다.",
            "result": None,
            "log": "",
        }
    EXECUTOR.submit(_execute_job, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
        return _clean(dict(job))


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
