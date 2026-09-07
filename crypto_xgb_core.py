"""범용 암호화폐 XGBoost 예측 엔진.

지원 대상은 Binance USD-M의 BTC/ETH/SOL 무기한 선물과
15분/1시간/4시간/1일 봉이다. 실제 주문 기능은 포함하지 않는다.

Model A 지표의 1, 3, 10, 12, 14, 20, 50, 200 등 모든 기간은 '일수'가 아니라
선택한 시간봉의 캔들 개수를 뜻한다. 예를 들어 EMA 200은 15분봉에서는
200×15분, 일봉에서는 200일을 의미한다.
"""

from __future__ import annotations

import json
import builtins
from contextvars import ContextVar
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ccxt
from market_data import BinancePublicData
import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

try:
    from filelock import FileLock, Timeout
except ImportError:
    # macOS/Linux에서는 추가 패키지가 없어도 안전하게 작동하도록 한다.
    # requirements_web.txt 설치 시에는 검증된 filelock 구현이 우선 사용된다.
    import fcntl

    class Timeout(Exception):
        """파일 잠금 대기시간 초과."""

    class FileLock:
        """filelock 미설치 시 사용하는 최소 POSIX 파일 잠금 구현."""

        def __init__(self, lock_file: str, timeout: float = -1) -> None:
            self.lock_file = lock_file
            self.timeout = timeout
            self._handle = None

        def __enter__(self):
            Path(self.lock_file).parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.lock_file, "a+", encoding="utf-8")
            deadline = (
                None if self.timeout < 0 else time.monotonic() + self.timeout
            )
            while True:
                try:
                    fcntl.flock(
                        self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    return self
                except BlockingIOError as exc:
                    if deadline is not None and time.monotonic() >= deadline:
                        self._handle.close()
                        self._handle = None
                        raise Timeout(self.lock_file) from exc
                    time.sleep(0.05)

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            if self._handle is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                self._handle.close()
                self._handle = None

try:
    from xgboost import XGBClassifier
except (ImportError, OSError) as exc:
    raise RuntimeError(
        "XGBoost를 불러오지 못했습니다. macOS에서는 `brew install libomp` 후 "
        "`python3 -m pip install -U xgboost`를 실행하세요."
    ) from exc


PROGRESS_CALLBACK = ContextVar("progress_callback", default=None)


def report(*args, **kwargs):
    callback = PROGRESS_CALLBACK.get()
    if callback is not None:
        callback(" ".join(str(arg) for arg in args))
    else:
        builtins.print(*args, **kwargs)


DISPLAY_TIMEZONE = "Asia/Seoul"
MODEL_NAME = "Model A"
MODEL_VERSION = "xgb_model_a_22_v1"
LIMIT_PER_CALL = 1000
MAX_FETCH_RETRIES = 3
RETRY_BASE_SECONDS = 1.0
PURGE_BARS = 1
TUNE_RATIO = 0.15
CALIBRATION_RATIO = 0.15
EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE = 42
DATA_LOCK_TIMEOUT_SECONDS = 30 * 60
MODEL_LOCK_TIMEOUT_SECONDS = 30 * 60
PAPER_LOCK_TIMEOUT_SECONDS = 60

LEVERAGE = 3.0
CAPITAL_FRACTION = 1.0
ONE_WAY_FEE_RATE = 0.0002
SLIPPAGE_RATE = 0.0
STARTING_CAPITAL_KRW = 1_000_000.0
THRESHOLDS = np.round(np.arange(0.52, 0.601, 0.01), 2)
MIN_THRESHOLD_TURNOVER = 20

SUPPORTED_SYMBOLS = {
    "BTC": "BTC/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
    "XRP": "XRP/USDT:USDT",
}


@dataclass(frozen=True)
class TimeframeConfig:
    timeframe: str
    display_name: str
    days: int
    initial_train_days: int
    test_days: int
    step_days: int
    delta: pd.Timedelta
    periods_per_year: int
    retrain_days: int


TIMEFRAME_CONFIGS = {
    "15m": TimeframeConfig(
        "15m", "15분", 365, 180, 30, 30, pd.Timedelta(minutes=15), 365 * 24 * 4, 1
    ),
    "1h": TimeframeConfig(
        "1h", "1시간", 365 * 3, 365, 60, 60, pd.Timedelta(hours=1), 365 * 24, 1
    ),
    "4h": TimeframeConfig(
        "4h", "4시간", 365 * 5, 365 * 2, 90, 90, pd.Timedelta(hours=4), 365 * 6, 7
    ),
    "1d": TimeframeConfig(
        "1d", "1일", 365 * 7, 365 * 2, 180, 180, pd.Timedelta(days=1), 365, 30
    ),
}

# Model A 피처 22개. 모든 기간은 선택한 시간봉의 캔들 개수이다.
FEATURE_COLUMNS = [
    # 모멘텀
    "return_1",
    "return_3",
    "return_12",
    "rsi_14",

    # 캔들 구조
    "range_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
    "streak_count",

    # 거래량
    "volume_change",
    "volume_zscore_20",
    "up_volume_ratio_10",

    # 변동성
    "volatility_20",
    "atr_14_ratio",
    "bb_pct",
    "parkinson_vol_20",

    # 추세
    "close_ema_50_ratio",
    "close_ema_200_ratio",

    # 시간 주기
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
]


@dataclass(frozen=True)
class XGBConfig:
    name: str
    max_depth: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    learning_rate: float
    reg_lambda: float
    reg_alpha: float
    n_estimators: int = 2000


XGB_CANDIDATES = [
    XGBConfig("X1", 3, 20, 0.70, 0.70, 0.03, 5.0, 0.0),
    XGBConfig("X2", 3, 40, 0.80, 0.70, 0.02, 10.0, 0.5),
    XGBConfig("X3", 4, 30, 0.70, 0.80, 0.02, 10.0, 1.0),
    XGBConfig("X4", 3, 60, 0.60, 0.60, 0.01, 20.0, 1.0),
]


@dataclass(frozen=True)
class StoragePaths:
    root: Path
    raw_ohlcv: Path
    raw_checkpoint: Path
    download_state: Path
    model: Path
    metadata: Path
    validation_dir: Path
    validation_predictions: Path
    validation_folds: Path
    validation_search: Path
    validation_summary: Path
    validation_reliability: Path
    validation_permutation: Path
    validation_noise: Path
    lockbox_predictions: Path
    lockbox_summary: Path
    paper_dir: Path
    paper_predictions: Path
    closed_trades: Path
    performance: Path


def _data_root() -> Path:
    override = os.environ.get("CRYPTO_XGB_DATA_ROOT")
    return Path(override).expanduser().resolve() if override else Path(__file__).resolve().parent / "data"


def symbol_slug(symbol: str) -> str:
    base_asset = symbol.split("/")[0].lower()
    return f"{base_asset}_usdt"


def symbol_from_token(token: str) -> str:
    """토큰명(예: BTC, 1000PEPE)을 CCXT USD-M 심볼로 변환한다."""
    normalized = str(token).strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,30}", normalized):
        raise ValueError(
            "토큰명은 영문 대문자와 숫자만 사용할 수 있습니다. 예: BTC, XRP, 1000PEPE"
        )
    return SUPPORTED_SYMBOLS.get(normalized, f"{normalized}/USDT:USDT")


def get_paths(symbol: str, timeframe: str, create: bool = True) -> StoragePaths:
    validate_selection(symbol, timeframe)
    root = _data_root() / symbol_slug(symbol) / timeframe
    validation_dir = root / "validation_model_a"
    paper_dir = root / "paper_trading_model_a"
    if create:
        validation_dir.mkdir(parents=True, exist_ok=True)
        paper_dir.mkdir(parents=True, exist_ok=True)
    return StoragePaths(
        root=root,
        raw_ohlcv=root / "raw_ohlcv.csv",
        raw_checkpoint=root / "raw_ohlcv.checkpoint.csv",
        download_state=root / "download_state.json",
        model=root / "model_a.joblib",
        metadata=root / "model_a_metadata.json",
        validation_dir=validation_dir,
        validation_predictions=validation_dir / "oos_predictions.csv",
        validation_folds=validation_dir / "fold_metrics.csv",
        validation_search=validation_dir / "parameter_search.csv",
        validation_summary=validation_dir / "summary.json",
        validation_reliability=validation_dir / "probability_reliability.csv",
        validation_permutation=validation_dir / "permutation_importance.csv",
        validation_noise=validation_dir / "noise_feature_test.csv",
        lockbox_predictions=validation_dir / "lockbox_predictions.csv",
        lockbox_summary=validation_dir / "lockbox_summary.json",
        paper_dir=paper_dir,
        paper_predictions=paper_dir / "predictions.csv",
        closed_trades=paper_dir / "closed_trades.csv",
        performance=paper_dir / "performance.json",
    )


def validate_selection(symbol: str, timeframe: str) -> None:
    if not re.fullmatch(r"[A-Z0-9]{2,30}/USDT:USDT", str(symbol).upper()):
        raise ValueError(f"올바르지 않은 USD-M USDT 심볼입니다: {symbol}")
    if timeframe not in TIMEFRAME_CONFIGS:
        raise ValueError(f"지원하지 않는 시간봉입니다: {timeframe}")


def timeframe_config(timeframe: str) -> TimeframeConfig:
    if timeframe not in TIMEFRAME_CONFIGS:
        raise ValueError(f"지원하지 않는 시간봉입니다: {timeframe}")
    return TIMEFRAME_CONFIGS[timeframe]


# ============================================================
# 데이터 수집과 검증
# ============================================================

RAW_CANDLE_COLUMNS = [
    "timestamp_ms", "open", "high", "low", "close", "volume"
]


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    """완전히 저장된 임시 파일만 정식 CSV 경로로 교체한다."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    """체크포인트 상태 JSON을 손상 없이 교체한다."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_joblib(payload: Any, destination: Path) -> None:
    """완전히 저장된 모델 파일만 정식 경로로 교체한다."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _fetch_page_with_retry(
    exchange,
    symbol: str,
    timeframe: str,
    since: int,
) -> list[list[float]]:
    """일시적인 네트워크·제한 오류에 지수 백오프로 재시도한다."""
    transient_errors = (
        ccxt.RequestTimeout,
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.RateLimitExceeded,
        ccxt.DDoSProtection,
    )
    last_error = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            return exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=LIMIT_PER_CALL,
            )
        except transient_errors as exc:
            last_error = exc
            if attempt == MAX_FETCH_RETRIES - 1:
                break
            delay = RETRY_BASE_SECONDS * (2 ** attempt)
            report(
                f"[API 재시도 {attempt + 1}/{MAX_FETCH_RETRIES - 1}] "
                f"{type(exc).__name__} | {delay:.0f}초 후 재시도"
            )
            time.sleep(delay)
    raise RuntimeError(
        f"OHLCV 요청이 {MAX_FETCH_RETRIES}회 연속 실패했습니다: {last_error}"
    ) from last_error


def _validate_exchange_market(exchange, symbol: str) -> None:
    """입력 토큰이 현재 거래 가능한 USDT 무기한 선물인지 확인한다."""
    transient_errors = (
        ccxt.RequestTimeout,
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
        ccxt.RateLimitExceeded,
        ccxt.DDoSProtection,
    )
    last_error = None
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            markets = exchange.load_markets()
            break
        except transient_errors as exc:
            last_error = exc
            if attempt == MAX_FETCH_RETRIES - 1:
                raise RuntimeError(
                    f"바이낸스 시장 목록 조회 실패: {last_error}"
                ) from exc
            time.sleep(RETRY_BASE_SECONDS * (2 ** attempt))

    market = markets.get(symbol)
    if market is None:
        token = symbol.split("/")[0]
        raise ValueError(
            f"{token}은(는) Binance USD-M의 USDT 선물에서 찾을 수 없습니다. "
            "현물에만 상장된 토큰일 수 있습니다."
        )
    if not market.get("swap") or not market.get("linear"):
        raise ValueError(f"{symbol}은(는) USDT 무기한 선물 계약이 아닙니다.")
    if market.get("quote") != "USDT" or market.get("settle") != "USDT":
        raise ValueError(f"{symbol}은(는) USDT 결제 선물 계약이 아닙니다.")
    if market.get("active") is False:
        raise ValueError(f"{symbol}은(는) 현재 거래가 중단된 계약입니다.")


def _raw_candle_frame(candles) -> pd.DataFrame:
    frame = pd.DataFrame(candles, columns=RAW_CANDLE_COLUMNS)
    if frame.empty:
        return frame
    frame["timestamp_ms"] = frame["timestamp_ms"].astype("int64")
    return (
        frame.drop_duplicates("timestamp_ms", keep="last")
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )


def _merge_raw_candles(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty:
        return right.copy()
    if right.empty:
        return left.copy()
    return (
        pd.concat([left, right], ignore_index=True)
        .drop_duplicates("timestamp_ms", keep="last")
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )


def _checkpoint_state_matches(
    state: dict[str, Any], symbol: str, timeframe: str, days: int
) -> bool:
    return (
        state.get("symbol") == symbol
        and state.get("timeframe") == timeframe
        and int(state.get("days", -1)) == int(days)
    )


def _save_download_checkpoint(
    frame: pd.DataFrame,
    paths: StoragePaths,
    symbol: str,
    timeframe: str,
    days: int,
    requested_start_ms: int,
) -> None:
    _atomic_write_csv(frame, paths.raw_checkpoint)
    _atomic_write_json(
        {
            "symbol": symbol,
            "timeframe": timeframe,
            "days": int(days),
            "requested_start_ms": int(requested_start_ms),
            "last_saved_timestamp_ms": int(frame["timestamp_ms"].iloc[-1]),
            "saved_candles": int(len(frame)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        paths.download_state,
    )


def _load_download_checkpoint(
    paths: StoragePaths, symbol: str, timeframe: str, days: int
) -> tuple[pd.DataFrame, int | None]:
    if not (paths.raw_checkpoint.exists() and paths.download_state.exists()):
        return pd.DataFrame(columns=RAW_CANDLE_COLUMNS), None
    try:
        with paths.download_state.open("r", encoding="utf-8") as file:
            state = json.load(file)
        if not _checkpoint_state_matches(state, symbol, timeframe, days):
            return pd.DataFrame(columns=RAW_CANDLE_COLUMNS), None
        frame = pd.read_csv(paths.raw_checkpoint)
        if frame.empty or not set(RAW_CANDLE_COLUMNS).issubset(frame.columns):
            return pd.DataFrame(columns=RAW_CANDLE_COLUMNS), None
        frame = _raw_candle_frame(frame[RAW_CANDLE_COLUMNS].to_numpy().tolist())
        requested_start_ms = int(state["requested_start_ms"])
        report(f"[수집 재개] 체크포인트 {len(frame):,}개에서 이어받습니다.")
        return frame, requested_start_ms
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return pd.DataFrame(columns=RAW_CANDLE_COLUMNS), None


def _finalize_raw_frame(
    raw: pd.DataFrame, timeframe_ms: int, current_ms: int
) -> pd.DataFrame:
    frame = raw.loc[raw["timestamp_ms"] + timeframe_ms <= current_ms].copy()
    if frame.empty:
        raise RuntimeError("종료된 캔들이 없습니다.")
    frame["timestamp"] = (
        pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
        .dt.tz_convert(DISPLAY_TIMEZONE)
    )
    return frame[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ].reset_index(drop=True)

def fetch_ohlcv(symbol: str, timeframe: str, days: int | None = None) -> pd.DataFrame:
    """종료 캔들을 체크포인트로 보존하며 전체 기간을 수집한다."""
    validate_selection(symbol, timeframe)
    config = timeframe_config(timeframe)
    days = config.days if days is None else int(days)
    paths = get_paths(symbol, timeframe)
    exchange = BinancePublicData()
    _validate_exchange_market(exchange, symbol)
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    requested_start_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
    )
    requested_start_ms = requested_start_ms // timeframe_ms * timeframe_ms
    raw, checkpoint_start_ms = _load_download_checkpoint(
        paths, symbol, timeframe, days
    )
    if checkpoint_start_ms is not None:
        requested_start_ms = checkpoint_start_ms
        # 마지막 체크포인트 봉은 당시 미완성일 수 있으므로 다시 요청한다.
        fetch_since = int(raw["timestamp_ms"].iloc[-1])
    else:
        raw = pd.DataFrame(columns=RAW_CANDLE_COLUMNS)
        fetch_since = requested_start_ms
    report(f"{symbol} {timeframe} 최근 {days}일 종료 캔들 수집 중...")

    while True:
        candles = _fetch_page_with_retry(
            exchange, symbol, timeframe, fetch_since
        )
        if not candles:
            break
        raw = _merge_raw_candles(raw, _raw_candle_frame(candles))
        _save_download_checkpoint(
            raw,
            paths,
            symbol,
            timeframe,
            days,
            requested_start_ms,
        )
        report(f"[수집 진행] {len(raw):,}개 캔들 저장")
        next_since = int(candles[-1][0]) + timeframe_ms
        if next_since <= fetch_since:
            break
        fetch_since = next_since
        if fetch_since >= exchange.milliseconds() or len(candles) < LIMIT_PER_CALL:
            break
        time.sleep(exchange.rateLimit / 1000)

    if raw.empty:
        raise RuntimeError(f"{symbol} {timeframe} 데이터를 가져오지 못했습니다.")
    df = _finalize_raw_frame(raw, timeframe_ms, exchange.milliseconds())
    validate_ohlcv(df, timeframe)
    _atomic_write_csv(df, paths.raw_ohlcv)
    paths.raw_checkpoint.unlink(missing_ok=True)
    paths.download_state.unlink(missing_ok=True)
    report(f"[완료] 종료 캔들 {len(df):,}개 -> {paths.raw_ohlcv}")
    return df


def update_ohlcv_incremental(symbol: str, timeframe: str) -> pd.DataFrame:
    """기존 CSV의 마지막 캔들 이후만 요청해 빠르게 갱신한다."""
    paths = get_paths(symbol, timeframe)
    if not paths.raw_ohlcv.exists():
        return fetch_ohlcv(symbol, timeframe)

    existing = pd.read_csv(paths.raw_ohlcv, parse_dates=["timestamp"])
    existing = existing.sort_values("timestamp").reset_index(drop=True)
    validate_ohlcv(existing, timeframe)
    exchange = BinancePublicData()
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    last_start_ms = int(pd.Timestamp(existing["timestamp"].iloc[-1]).timestamp() * 1000)
    fetch_since = last_start_ms + timeframe_ms
    current_ms = exchange.milliseconds()

    if fetch_since + timeframe_ms > current_ms:
        report(f"[데이터 최신] 종료 캔들 {len(existing):,}개 | 신규 캔들 없음")
        return existing

    _validate_exchange_market(exchange, symbol)
    report(
        f"{symbol} {timeframe} 마지막 저장 캔들 이후 증분 데이터 수집 중..."
    )
    new_candles = []
    while fetch_since < current_ms:
        candles = _fetch_page_with_retry(
            exchange, symbol, timeframe, fetch_since
        )
        if not candles:
            break
        new_candles.extend(candles)
        next_since = int(candles[-1][0]) + timeframe_ms
        if next_since <= fetch_since:
            break
        fetch_since = next_since
        if len(candles) < LIMIT_PER_CALL:
            break
        time.sleep(exchange.rateLimit / 1000)

    if new_candles:
        new_df = pd.DataFrame(
            new_candles,
            columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
        )
        new_df = new_df.loc[
            new_df["timestamp_ms"] + timeframe_ms <= exchange.milliseconds()
        ].copy()
        new_df["timestamp"] = (
            pd.to_datetime(new_df["timestamp_ms"], unit="ms", utc=True)
            .dt.tz_convert(DISPLAY_TIMEZONE)
        )
        new_df = new_df[
            ["timestamp", "open", "high", "low", "close", "volume"]
        ]
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = (
            combined.drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
    else:
        combined = existing

    validate_ohlcv(combined, timeframe)
    _atomic_write_csv(combined, paths.raw_ohlcv)
    added = len(combined) - len(existing)
    report(f"[갱신 완료] 신규 종료 캔들 {added:,}개 | 전체 {len(combined):,}개")
    return combined


def load_raw_data(symbol: str, timeframe: str, refresh: bool = True) -> pd.DataFrame:
    paths = get_paths(symbol, timeframe)
    # Writers use atomic replace, so a reader can safely use the last complete
    # snapshot without waiting behind a potentially long download lock.
    if not refresh and paths.raw_ohlcv.exists():
        df = pd.read_csv(paths.raw_ohlcv, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        validate_ohlcv(df, timeframe)
        return df
    data_lock = FileLock(
        str(paths.raw_ohlcv) + ".lock",
        timeout=DATA_LOCK_TIMEOUT_SECONDS,
    )
    try:
        with data_lock:
            if refresh and paths.raw_ohlcv.exists():
                return update_ohlcv_incremental(symbol, timeframe)
            if not paths.raw_ohlcv.exists():
                return fetch_ohlcv(symbol, timeframe)
            df = pd.read_csv(paths.raw_ohlcv, parse_dates=["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            validate_ohlcv(df, timeframe)
            return df
    except Timeout as exc:
        raise RuntimeError(
            "동일한 코인·시간봉의 OHLCV 데이터를 다른 프로세스가 갱신 중입니다. "
            "잠시 후 다시 실행하세요."
        ) from exc


def import_raw_csv(source: str | Path, symbol: str, timeframe: str) -> pd.DataFrame:
    """기존 CSV를 범용 조합별 저장경로로 가져온다. 원본은 수정하지 않는다."""
    df = pd.read_csv(source, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    validate_ohlcv(df, timeframe)
    paths = get_paths(symbol, timeframe)
    data_lock = FileLock(
        str(paths.raw_ohlcv) + ".lock",
        timeout=DATA_LOCK_TIMEOUT_SECONDS,
    )
    try:
        with data_lock:
            _atomic_write_csv(df, paths.raw_ohlcv)
    except Timeout as exc:
        raise RuntimeError(
            "동일한 코인·시간봉의 OHLCV 데이터를 다른 프로세스가 갱신 중입니다. "
            "잠시 후 다시 실행하세요."
        ) from exc
    return df


def validate_ohlcv(df: pd.DataFrame, timeframe: str) -> None:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV 필수 열 누락: {sorted(missing)}")
    if df.empty:
        raise ValueError("OHLCV 데이터가 비어 있습니다.")
    values = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    if df["timestamp"].isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("OHLCV에 결측값 또는 유효하지 않은 숫자가 있습니다.")
    if ((values["high"] < values[["open", "close", "low"]].max(axis=1)) |
            (values["low"] > values[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("OHLCV 고가·저가 범위가 올바르지 않습니다.")
    if df["timestamp"].duplicated().any():
        raise ValueError("중복 timestamp가 존재합니다.")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("OHLCV가 시간순으로 정렬되지 않았습니다.")
    expected = timeframe_config(timeframe).delta
    bad_gap = df["timestamp"].diff().iloc[1:] != expected
    if bad_gap.any():
        examples = df.loc[bad_gap.index[bad_gap], "timestamp"].head(5).tolist()
        raise ValueError(f"{timeframe} 간격이 아닌 캔들이 존재합니다: {examples}")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("0 이하 가격이 존재합니다.")
    if (df["volume"] < 0).any():
        raise ValueError("음수 거래량이 존재합니다.")


# ============================================================
# Model A 피처
# ============================================================

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """t까지의 정보로 22개 피처를 만들고 t+1 종가 방향을 타깃으로 만든다."""
    result = df.copy()
    open_price, high, low = result["open"], result["high"], result["low"]
    close, volume = result["close"], result["volume"]
    safe_open = open_price.replace(0, np.nan)
    safe_close = close.replace(0, np.nan)

    # 모멘텀
    result["return_1"] = close.pct_change(1, fill_method=None)
    result["return_3"] = close.pct_change(3, fill_method=None)
    result["return_12"] = close.pct_change(12, fill_method=None)
    log_return = np.log(close / close.shift(1))

    price_delta = close.diff()
    gain = price_delta.clip(lower=0)
    loss = -price_delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi_denominator = (avg_gain + avg_loss).replace(0, np.nan)
    result["rsi_14"] = 100 * avg_gain / rsi_denominator

    # 캔들 구조
    candle_top = pd.concat([open_price, close], axis=1).max(axis=1)
    candle_bottom = pd.concat([open_price, close], axis=1).min(axis=1)
    candle_range = (high - low).replace(0, np.nan)
    result["range_ratio"] = (high - low) / safe_open
    result["upper_wick_ratio"] = (high - candle_top) / safe_open
    result["lower_wick_ratio"] = (candle_bottom - low) / safe_open
    result["close_position"] = (close - low) / candle_range

    direction = np.sign(price_delta).fillna(0).astype(int)
    direction_group = direction.ne(direction.shift()).cumsum()
    run_length = direction.groupby(direction_group).cumcount() + 1
    result["streak_count"] = (run_length * direction).clip(-10, 10).astype(float)

    # 거래량
    result["volume_change"] = volume.pct_change(fill_method=None)
    volume_mean_20 = volume.rolling(20, min_periods=20).mean()
    volume_std_20 = volume.rolling(20, min_periods=20).std().replace(0, np.nan)
    result["volume_zscore_20"] = (volume - volume_mean_20) / volume_std_20
    up_volume = volume.where(close > open_price, 0.0)
    result["up_volume_ratio_10"] = (
        up_volume.rolling(10, min_periods=10).sum()
        / volume.rolling(10, min_periods=10).sum().replace(0, np.nan)
    )

    # 변동성
    result["volatility_20"] = log_return.rolling(20, min_periods=20).std()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    result["atr_14_ratio"] = atr_14 / safe_close

    close_sma_20 = close.rolling(20, min_periods=20).mean()
    close_std_20 = close.rolling(20, min_periods=20).std(ddof=0)
    bb_upper = close_sma_20 + 2 * close_std_20
    bb_lower = close_sma_20 - 2 * close_std_20
    result["bb_pct"] = (
        (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    )

    log_high_low_squared = np.log(high / low.replace(0, np.nan)) ** 2
    result["parkinson_vol_20"] = np.sqrt(
        log_high_low_squared.rolling(20, min_periods=20).mean()
        / (4 * np.log(2))
    )

    # 추세
    for period in [50, 200]:
        ema = _ema(close, period)
        result[f"close_ema_{period}_ratio"] = close / ema - 1

    # 시간 주기
    hour = result["timestamp"].dt.hour
    day = result["timestamp"].dt.dayofweek
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_sin"] = np.sin(2 * np.pi * day / 7)
    result["day_cos"] = np.cos(2 * np.pi * day / 7)

    result["future_return"] = close.shift(-1) / close - 1
    result["target_up"] = np.where(
        result["future_return"].notna(),
        (result["future_return"] > 0).astype(float),
        np.nan,
    )
    return result.replace([np.inf, -np.inf], np.nan)


def make_model_data(featured: pd.DataFrame) -> pd.DataFrame:
    required = ["timestamp", "close", *FEATURE_COLUMNS, "future_return", "target_up"]
    result = (
        featured[required]
        .dropna(subset=FEATURE_COLUMNS + ["future_return", "target_up"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if len(result) < 400:
        raise ValueError(
            f"학습 가능 데이터가 {len(result):,}개뿐입니다. 최소 400개가 필요합니다."
        )
    return result


def prepare_data(symbol: str, timeframe: str, refresh: bool = True):
    raw = load_raw_data(symbol, timeframe, refresh=refresh)
    featured = add_features(raw)
    model_df = make_model_data(featured)
    return raw, featured, model_df


# ============================================================
# XGBoost 튜닝과 확률 보정
# ============================================================

def make_model(config: XGBConfig, early_stopping: bool) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_depth=config.max_depth,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        reg_lambda=config.reg_lambda,
        reg_alpha=config.reg_alpha,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS if early_stopping else None,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def split_selection_data(model_df: pd.DataFrame):
    """fit→tune→calibration-fit→calibration-select 순서로 분리한다."""
    n = len(model_df)
    calibration_start = n - int(n * CALIBRATION_RATIO)
    tune_start = calibration_start - int(n * TUNE_RATIO)
    fit_df = model_df.iloc[: tune_start - PURGE_BARS].copy()
    tune_df = model_df.iloc[tune_start : calibration_start - PURGE_BARS].copy()
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
    if min(sizes.values()) < 50:
        raise ValueError(f"학습 내부 구간이 부족합니다: {sizes}")
    return fit_df, tune_df, calibration_fit, calibration_select


def probability_metrics(y_true, probability) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def tune_model(
    fit_df: pd.DataFrame,
    tune_df: pd.DataFrame,
    fold: int = 0,
    verbose: bool = False,
):
    rows = []
    y_tune = tune_df["target_up"].astype(int)
    for config in XGB_CANDIDATES:
        if verbose:
            report(
                f"[모델 튜닝] {config.name} | depth={config.max_depth}, "
                f"learning_rate={config.learning_rate}"
            )
        model = make_model(config, early_stopping=True)
        model.fit(
            fit_df[FEATURE_COLUMNS],
            fit_df["target_up"].astype(int),
            eval_set=[(tune_df[FEATURE_COLUMNS], y_tune)],
            verbose=False,
        )
        p = model.predict_proba(tune_df[FEATURE_COLUMNS])[:, 1]
        rows.append(
            {
                "fold": fold,
                "config": config.name,
                "best_trees": int(model.best_iteration) + 1,
                **probability_metrics(y_tune, p),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["roc_auc", "log_loss"], ascending=[False, True]
    ).reset_index(drop=True)
    winner = table.iloc[0]
    original = next(item for item in XGB_CANDIDATES if item.name == winner["config"])
    selected = XGBConfig(
        **{**asdict(original), "n_estimators": int(winner["best_trees"])}
    )
    if verbose:
        report(
            f"[모델 선택] {selected.name} | 최적 트리 {selected.n_estimators}개"
        )
    return selected, table


def _logit(probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibrators(y_true, raw_probability) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    raw = np.asarray(raw_probability, dtype=float)
    sigmoid = LogisticRegression(C=1e6, l1_ratio=0.0, max_iter=2000)
    sigmoid.fit(_logit(raw).reshape(-1, 1), y)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw, y)
    return {"raw": None, "sigmoid": sigmoid, "isotonic": isotonic}


def apply_calibrator(name: str, calibrator, raw_probability) -> np.ndarray:
    raw = np.asarray(raw_probability, dtype=float)
    if name == "raw":
        result = raw
    elif name == "sigmoid":
        result = calibrator.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
    elif name == "isotonic":
        result = calibrator.predict(raw)
    else:
        raise ValueError(f"알 수 없는 확률 보정법: {name}")
    return np.clip(result, 1e-6, 1 - 1e-6)


def select_calibrator(calibrators, y_select, raw_select):
    """보정기 학습에 사용하지 않은 다음 구간에서 Brier로 선택한다."""
    y = np.asarray(y_select, dtype=int)
    candidates = []
    for name, calibrator in calibrators.items():
        p = apply_calibrator(name, calibrator, raw_select)
        candidates.append(
            {
                "name": name,
                "calibrator": calibrator,
                "brier": float(brier_score_loss(y, p)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
            }
        )
    selected = min(candidates, key=lambda row: (row["brier"], row["log_loss"]))
    return selected, [{k: v for k, v in row.items() if k != "calibrator"} for row in candidates]


def positions_from_probability(probability, threshold) -> np.ndarray:
    p = np.asarray(probability, dtype=float)
    t = np.asarray(threshold, dtype=float)
    return np.select([p >= t, p <= 1 - t], [1.0, -1.0], default=0.0)


def _threshold_score(probability, future_return, threshold: float) -> dict[str, float]:
    position = positions_from_probability(probability, threshold)
    previous = np.r_[0.0, position[:-1]]
    turnover = np.abs(position - previous)
    if len(turnover):
        turnover[-1] += abs(position[-1])
    net = position * np.asarray(future_return) - ONE_WAY_FEE_RATE * turnover
    std = net.std(ddof=1)
    sharpe = net.mean() / std if std > 0 else np.nan
    equity = np.cumprod(1 + net)
    return {
        "threshold": float(threshold),
        "sharpe_unannualized": float(sharpe),
        "total_return": float(equity[-1] - 1),
        "turnover_units": float(turnover.sum()),
    }


def choose_threshold(probability, future_return):
    rows = [
        _threshold_score(probability, future_return, float(t)) for t in THRESHOLDS
    ]
    eligible = [
        row for row in rows
        if row["turnover_units"] >= MIN_THRESHOLD_TURNOVER
        and np.isfinite(row["sharpe_unannualized"])
    ]
    if not eligible:
        return 0.56, rows
    eligible.sort(
        key=lambda row: (row["sharpe_unannualized"], row["total_return"]),
        reverse=True,
    )
    return float(eligible[0]["threshold"]), rows


def fit_operational_artifact(
    model_df: pd.DataFrame,
    latest_source_timestamp,
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    fit_df, tune_df, calibration_fit, calibration_select = split_selection_data(model_df)
    report("[모델 학습] 저장된 운영 모델이 없거나 재학습 주기에 도달했습니다.")
    config, parameter_search = tune_model(fit_df, tune_df, verbose=True)
    pre_calibration = pd.concat([fit_df, tune_df], ignore_index=True)
    model = make_model(config, early_stopping=False)
    model.fit(pre_calibration[FEATURE_COLUMNS], pre_calibration["target_up"].astype(int))

    p_fit_raw = model.predict_proba(calibration_fit[FEATURE_COLUMNS])[:, 1]
    p_select_raw = model.predict_proba(calibration_select[FEATURE_COLUMNS])[:, 1]
    calibrators = fit_calibrators(calibration_fit["target_up"], p_fit_raw)
    selected, calibration_search = select_calibrator(
        calibrators, calibration_select["target_up"], p_select_raw
    )
    p_select = apply_calibrator(
        selected["name"], selected["calibrator"], p_select_raw
    )
    threshold, threshold_search = choose_threshold(
        p_select, calibration_select["future_return"].to_numpy()
    )
    trained_at = datetime.now(timezone.utc).isoformat()
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model": model,
        "calibrator": selected["calibrator"],
        "calibration_method": selected["name"],
        "threshold": threshold,
        "xgb_config": asdict(config),
        "feature_columns": FEATURE_COLUMNS,
        "symbol": symbol,
        "timeframe": timeframe,
        "trained_at": trained_at,
        "latest_source_timestamp": str(latest_source_timestamp),
        "available_labeled_end": str(model_df["timestamp"].iloc[-1]),
        "available_labeled_samples": len(model_df),
        "parameter_search": parameter_search.to_dict(orient="records"),
        "calibration_search": calibration_search,
        "threshold_search": threshold_search,
    }


def save_artifact(artifact: dict[str, Any], paths: StoragePaths) -> None:
    _atomic_write_joblib(artifact, paths.model)
    metadata = {
        key: value for key, value in artifact.items() if key not in {"model", "calibrator"}
    }
    metadata.update(
        {
            "leverage": LEVERAGE,
            "capital_fraction": CAPITAL_FRACTION,
            "one_way_fee_on_notional": ONE_WAY_FEE_RATE,
            "one_way_fee_on_equity": LEVERAGE * CAPITAL_FRACTION * ONE_WAY_FEE_RATE,
            "funding_included": False,
            "actual_orders_enabled": False,
        }
    )
    _atomic_write_json(metadata, paths.metadata)


def load_artifact(paths: StoragePaths) -> dict[str, Any] | None:
    if not paths.model.exists():
        return None
    try:
        artifact = joblib.load(paths.model)
        return artifact if isinstance(artifact, dict) else None
    except Exception as exc:
        report(f"[모델 재학습] 저장 모델을 읽지 못했습니다: {type(exc).__name__}")
        return None


def artifact_is_current(artifact: dict | None, latest_timestamp, symbol, timeframe) -> bool:
    if artifact is None:
        return False
    if artifact.get("model_version") != MODEL_VERSION:
        return False
    if artifact.get("feature_columns") != FEATURE_COLUMNS:
        return False
    if artifact.get("symbol") != symbol or artifact.get("timeframe") != timeframe:
        return False
    trained_at = pd.to_datetime(artifact.get("trained_at"), utc=True, errors="coerce")
    if pd.isna(trained_at):
        return False
    age = pd.Timestamp.now(tz="UTC") - trained_at
    return age < pd.Timedelta(days=timeframe_config(timeframe).retrain_days)


def get_or_train_artifact(
    model_df: pd.DataFrame,
    latest_timestamp,
    symbol: str,
    timeframe: str,
    force_retrain: bool = False,
):
    paths = get_paths(symbol, timeframe)
    model_lock = FileLock(
        str(paths.model) + ".lock",
        timeout=MODEL_LOCK_TIMEOUT_SECONDS,
    )
    try:
        with model_lock:
            # 잠금을 얻은 뒤 다시 확인해야 다른 프로세스가 방금 저장한 모델을
            # 불필요하게 재학습하지 않는다.
            artifact = load_artifact(paths)
            reused = not force_retrain and artifact_is_current(
                artifact, latest_timestamp, symbol, timeframe
            )
            if not reused:
                artifact = fit_operational_artifact(
                    model_df, latest_timestamp, symbol, timeframe
                )
                save_artifact(artifact, paths)
            return artifact, reused
    except Timeout as exc:
        raise RuntimeError(
            "동일한 코인·시간봉의 모델을 다른 프로세스가 학습 중입니다. "
            "잠시 후 다시 실행하세요."
        ) from exc


# ============================================================
# 최신 예측, 중복 방지, 이전 예측 정산
# ============================================================

PAPER_COLUMNS = [
    "prediction_created_at",
    "source_candle_start",
    "source_candle_end",
    "target_candle_start",
    "target_candle_end",
    "reference_close",
    "raw_probability_up",
    "probability_up",
    "probability_down",
    "calibration_method",
    "threshold",
    "signal",
    "signal_text",
    "status",
    "realized_close",
    "asset_return",
    "net_strategy_return",
    "fee_return",
]


def load_paper_log(paths: StoragePaths) -> pd.DataFrame:
    if not paths.paper_predictions.exists():
        return pd.DataFrame(columns=PAPER_COLUMNS)
    frame = pd.read_csv(paths.paper_predictions)
    for column in [
        "source_candle_start", "source_candle_end",
        "target_candle_start", "target_candle_end",
    ]:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column])
    for column in PAPER_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame[PAPER_COLUMNS]


def settle_paper_log(log_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return log_df
    lookup = raw_df.set_index("timestamp")
    effective_leverage = LEVERAGE * CAPITAL_FRACTION
    for index, row in log_df.loc[log_df["status"] == "pending"].iterrows():
        target_start = pd.Timestamp(row["target_candle_start"])
        if target_start not in lookup.index:
            continue
        candle = lookup.loc[target_start]
        asset_return = float(candle["close"] / float(row["reference_close"]) - 1)
        signal = int(row["signal"])
        fee = float(row["fee_return"])
        log_df.loc[index, "status"] = "settled"
        log_df.loc[index, "realized_close"] = float(candle["close"])
        log_df.loc[index, "asset_return"] = asset_return
        log_df.loc[index, "net_strategy_return"] = (
            effective_leverage * signal * asset_return - fee
        )
    return log_df


def _signal(probability: float, threshold: float) -> tuple[int, str]:
    if probability >= threshold:
        return 1, "LONG"
    if probability <= 1 - threshold:
        return -1, "SHORT"
    return 0, "NEUTRAL"


def append_prediction(log_df, latest_row, artifact, config: TimeframeConfig):
    source_start = pd.Timestamp(latest_row["timestamp"].iloc[0])
    source_end = source_start + config.delta
    target_start = source_end
    target_end = target_start + config.delta
    existing = set(pd.to_datetime(log_df["target_candle_start"])) if not log_df.empty else set()
    if target_start in existing:
        return log_df, False

    raw_probability = float(
        artifact["model"].predict_proba(latest_row[FEATURE_COLUMNS])[0, 1]
    )
    probability = float(
        apply_calibrator(
            artifact["calibration_method"],
            artifact["calibrator"],
            np.asarray([raw_probability]),
        )[0]
    )
    threshold = float(artifact["threshold"])
    signal, signal_text = _signal(probability, threshold)
    previous_signal = int(log_df.iloc[-1]["signal"]) if not log_df.empty else 0
    turnover = abs(signal - previous_signal)
    fee = LEVERAGE * CAPITAL_FRACTION * ONE_WAY_FEE_RATE * turnover
    row = {column: np.nan for column in PAPER_COLUMNS}
    row.update(
        {
            "prediction_created_at": datetime.now(timezone.utc).isoformat(),
            "source_candle_start": source_start,
            "source_candle_end": source_end,
            "target_candle_start": target_start,
            "target_candle_end": target_end,
            "reference_close": float(latest_row["close"].iloc[0]),
            "raw_probability_up": raw_probability,
            "probability_up": probability,
            "probability_down": 1 - probability,
            "calibration_method": artifact["calibration_method"],
            "threshold": threshold,
            "signal": signal,
            "signal_text": signal_text,
            "status": "pending",
            "fee_return": fee,
        }
    )
    return pd.concat([log_df, pd.DataFrame([row])], ignore_index=True), True


def run_latest_prediction(
    symbol: str,
    timeframe: str,
    refresh: bool = True,
    force_retrain: bool = False,
) -> dict[str, Any]:
    raw, featured, model_df = prepare_data(symbol, timeframe, refresh=refresh)
    latest = featured.dropna(subset=FEATURE_COLUMNS).sort_values("timestamp").tail(1)
    if latest.empty:
        raise ValueError("최신 예측용 피처가 없습니다.")
    latest_timestamp = latest["timestamp"].iloc[0]
    artifact, reused = get_or_train_artifact(
        model_df, latest_timestamp, symbol, timeframe, force_retrain=force_retrain
    )
    paths = get_paths(symbol, timeframe)
    paper_lock = FileLock(
        str(paths.paper_predictions) + ".lock",
        timeout=PAPER_LOCK_TIMEOUT_SECONDS,
    )
    try:
        with paper_lock:
            # 읽기→정산→중복 검사→추가→저장의 전체 과정을 하나의 임계구역으로
            # 묶어 동시 실행 시 기록 유실과 중복 행 생성을 막는다.
            log_df = load_paper_log(paths)
            log_df = settle_paper_log(log_df, raw)
            log_df, created = append_prediction(
                log_df, latest, artifact, timeframe_config(timeframe)
            )
            _atomic_write_csv(log_df, paths.paper_predictions)
            latest_log = (
                log_df.sort_values("target_candle_start").iloc[-1].copy()
            )
    except Timeout as exc:
        raise RuntimeError(
            "동일한 코인·시간봉의 예측 기록을 다른 프로세스가 갱신 중입니다. "
            "잠시 후 다시 실행하세요."
        ) from exc
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "source_candle_start": latest_log["source_candle_start"],
        "source_candle_end": latest_log["source_candle_end"],
        "target_candle_start": latest_log["target_candle_start"],
        "target_candle_end": latest_log["target_candle_end"],
        "reference_close": float(latest_log["reference_close"]),
        "probability_up": float(latest_log["probability_up"]),
        "probability_down": float(latest_log["probability_down"]),
        "calibration_method": latest_log["calibration_method"],
        "threshold": float(latest_log["threshold"]),
        "signal": latest_log["signal_text"],
        "trained_at": artifact["trained_at"],
        "model_reused": reused,
        "prediction_created": created,
        "paths": paths,
    }
