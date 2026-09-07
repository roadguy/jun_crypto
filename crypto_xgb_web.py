"""Streamlit UI for the existing cryptocurrency XGBoost Model A engine.

This module deliberately keeps data collection, feature engineering, model
training, calibration, threshold selection, paper logging, and walk-forward
validation inside ``crypto_xgb_core`` and ``crypto_xgb_validation``.  It only
coordinates explicit button actions and renders their saved results.

Run with::

    python3 -m streamlit run crypto_xgb_web.py

No order-placement API is present in this file.
"""

from __future__ import annotations

import html
import io
import json
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import crypto_xgb_core as core
import crypto_xgb_validation as validation
import crypto_xgb_range_live as range_live


TIMEFRAME_LABELS = {
    "15m": "15분",
    "1h": "1시간",
    "4h": "4시간",
    "1d": "1일",
}
SESSION_RESULTS_KEY = "crypto_xgb_web_results"


def normalize_selection(token: str, timeframe: str) -> tuple[str, str, str]:
    """Normalize a token and return (token, CCXT symbol, timeframe)."""
    normalized = str(token).strip().upper()
    if not normalized:
        raise ValueError("코인 토큰명을 입력하세요. 예: BTC, ETH, XRP")
    symbol = core.symbol_from_token(normalized)
    core.validate_selection(symbol, timeframe)
    return normalized, symbol, timeframe


def selection_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}|{timeframe}"


def format_timestamp(value: Any, include_timezone: bool = True) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "알 수 없음"
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(core.DISPLAY_TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(core.DISPLAY_TIMEZONE)
    pattern = "%Y-%m-%d %H:%M:%S%z" if include_timezone else "%Y-%m-%d %H:%M"
    text = timestamp.strftime(pattern)
    if include_timezone and len(text) >= 5:
        text = f"{text[:-2]}:{text[-2:]}"
    return text


def metric_text(value: Any, formatter, unavailable: str = "계산 불가") -> str:
    if value is None:
        return unavailable
    try:
        number = float(value)
    except (TypeError, ValueError):
        return unavailable
    if pd.isna(number):
        return unavailable
    return formatter(number)


def friendly_error(exc: Exception) -> str:
    """Convert expected operational failures into concise Korean messages."""
    message = str(exc).strip() or type(exc).__name__
    lowered = message.lower()
    if isinstance(exc, ValueError):
        return message
    if "xgboost" in lowered or "libomp" in lowered:
        return "XGBoost 실행 환경을 불러오지 못했습니다. 설치 안내를 확인하세요."
    if "ohlcv" in lowered or "캔들" in message or "데이터" in message:
        return f"가격 데이터를 처리하지 못했습니다: {message}"
    if any(word in lowered for word in ("timeout", "network", "exchange", "request")):
        return "Binance 공개 데이터 연결에 실패했습니다. 잠시 후 다시 시도하세요."
    if isinstance(exc, (OSError, json.JSONDecodeError)):
        return "저장된 데이터 또는 모델 파일을 읽지 못했습니다."
    return f"요청을 완료하지 못했습니다: {message}"


def show_error(exc: Exception, operation_log: str = "") -> None:
    st.error(friendly_error(exc))
    details = f"{type(exc).__name__}: {exc}"
    if operation_log.strip():
        details += "\n\n[실행 기록]\n" + operation_log.strip()
    with st.expander("오류 상세 정보"):
        st.code(details)


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def recent_ohlcv(raw_df: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    result = raw_df.sort_values("timestamp").tail(rows)[columns].copy()
    result["timestamp"] = result["timestamp"].map(format_timestamp)
    return result


def candlestick_figure(
    raw_df: pd.DataFrame,
    prediction: dict[str, Any] | None = None,
) -> go.Figure:
    """거래소형 가격 차트.

    저장된 모든 종료 캔들을 trace에 넣되, 최초 화면은 최근 100개로 제한한다.
    따라서 사용자는 휠 확대·축소와 드래그로 과거 전체 구간을 탐색할 수 있다.
    """
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    chart = raw_df.loc[:, columns].copy()
    chart["timestamp"] = pd.to_datetime(chart["timestamp"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        chart[column] = pd.to_numeric(chart[column], errors="coerce")
    chart = (
        chart.dropna(subset=columns)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    if chart.empty:
        raise ValueError("차트에 표시할 종료 캔들이 없습니다.")

    # 지표는 화면에 보이는 100개가 아니라 전체 이력에서 계산한다.
    close = chart["close"]
    chart["ema_50"] = close.ewm(span=50, adjust=False).mean()
    chart["ema_200"] = close.ewm(span=200, adjust=False).mean()
    chart["bb_mid"] = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std(ddof=0)
    chart["bb_upper"] = chart["bb_mid"] + 2.0 * bb_std
    chart["bb_lower"] = chart["bb_mid"] - 2.0 * bb_std

    timestamps = chart["timestamp"]
    candle_width = (
        timestamps.iloc[-1] - timestamps.iloc[-2]
        if len(timestamps) >= 2
        else pd.Timedelta(hours=1)
    )
    zone_start = timestamps.iloc[-1] + candle_width * 0.7
    zone_end = timestamps.iloc[-1] + candle_width * 10
    visible = chart.tail(100)
    low = float(visible["low"].min())
    high = float(visible["high"].max())
    padding = max((high - low) * 0.05, abs(high) * 0.001)
    y_min, y_max = low - padding, high + padding
    current_price = float(chart["close"].iloc[-1])
    probability_up = (
        float(prediction["probability_up"]) if prediction is not None else None
    )
    probability_down = 1.0 - probability_up if probability_up is not None else None
    signal = str(prediction.get("signal", "")).upper() if prediction else ""
    figure = go.Figure()

    # Bollinger 하단을 먼저 그리고 상단 trace의 fill로 밴드 영역을 만든다.
    figure.add_trace(
        go.Scatter(
            x=chart["timestamp"], y=chart["bb_lower"],
            mode="lines", line=dict(color="rgba(147, 112, 219, 0.62)", width=1),
            name="BB Lower", hoverinfo="skip", legendgroup="bb",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=chart["timestamp"], y=chart["bb_upper"],
            mode="lines", line=dict(color="rgba(147, 112, 219, 0.62)", width=1),
            fill="tonexty", fillcolor="rgba(147, 112, 219, 0.075)",
            name="Bollinger 20·2", hoverinfo="skip", legendgroup="bb",
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=chart["timestamp"], y=chart["ema_50"], mode="lines",
            line=dict(color="#e3b854", width=1.25), name="EMA 50",
            hovertemplate="EMA 50 %{y:,.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scattergl(
            x=chart["timestamp"], y=chart["ema_200"], mode="lines",
            line=dict(color="#65a9ff", width=1.25), name="EMA 200",
            hovertemplate="EMA 200 %{y:,.2f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Candlestick(
            x=chart["timestamp"],
            open=chart["open"], high=chart["high"],
            low=chart["low"], close=chart["close"],
            increasing_line_color="#4fd1ae", increasing_fillcolor="#4fd1ae",
            decreasing_line_color="#ef6461", decreasing_fillcolor="#ef6461",
            name="OHLC",
            hovertext=[
                f"O {o:,.2f}<br>H {h:,.2f}<br>L {l:,.2f}<br>C {c:,.2f}"
                for o, h, l, c in zip(
                    chart["open"], chart["high"], chart["low"], chart["close"]
                )
            ],
        )
    )

    # 신호 영역은 마지막 캔들 오른쪽의 미래 공간에만 둔다. 가격 목표 범위가
    # 아니라 방향분류 확률을 보여주는 장식 패널이라는 점을 분리해 표현한다.
    figure.add_shape(
        type="rect", x0=zone_start, x1=zone_end,
        y0=current_price, y1=y_max,
        line=dict(width=0), fillcolor="rgba(79, 209, 174, 0.09)", layer="below",
    )
    figure.add_shape(
        type="rect", x0=zone_start, x1=zone_end,
        y0=y_min, y1=current_price,
        line=dict(width=0), fillcolor="rgba(239, 100, 97, 0.09)", layer="below",
    )
    figure.add_shape(
        type="line", x0=zone_start, x1=zone_end,
        y0=current_price, y1=current_price,
        line=dict(color="#c9a24b", width=2, dash="dot"),
    )
    long_label = "LONG"
    short_label = "SHORT"
    if probability_up is not None:
        long_label += f"  {probability_up:.1%}"
        short_label += f"  {probability_down:.1%}"
    label_x = zone_start + (zone_end - zone_start) * 0.53
    figure.add_annotation(
        x=label_x, y=current_price + (y_max - current_price) * 0.54,
        text=f"<b>{long_label}</b>", showarrow=False,
        font=dict(color="#4fd1ae", size=11, family="JetBrains Mono, monospace"),
    )
    figure.add_annotation(
        x=label_x, y=current_price - (current_price - y_min) * 0.54,
        text=f"<b>{short_label}</b>", showarrow=False,
        font=dict(color="#ef6461", size=11, family="JetBrains Mono, monospace"),
    )
    figure.add_annotation(
        x=label_x, y=current_price,
        text="<b>NEUTRAL</b>" if signal == "NEUTRAL" else "NEUTRAL",
        showarrow=False, yshift=11,
        font=dict(color="#c9a24b", size=10, family="JetBrains Mono, monospace"),
    )

    # 실제 마지막 종가 라인과 우측 가격 태그. 신호 영역의 NEUTRAL 선과
    # 구분되도록 전체 가격 패널을 가로지르는 별도 회색 실선으로 표시한다.
    figure.add_hline(
        y=current_price,
        line=dict(color="rgba(225, 230, 236, 0.72)", width=1, dash="solid"),
    )
    figure.add_annotation(
        x=1.0, xref="paper", xanchor="left", xshift=4,
        y=current_price, yref="y", yanchor="middle",
        text=f"<b>{current_price:,.2f}</b>", showarrow=False,
        bgcolor="#dfe5ec", bordercolor="#dfe5ec", borderpad=4,
        font=dict(color="#111820", size=10, family="JetBrains Mono, monospace"),
    )

    initial_start = timestamps.iloc[max(0, len(timestamps) - 100)]
    figure.update_layout(
        height=560,
        margin=dict(l=8, r=72, t=50, b=8),
        hovermode="closest",
        dragmode="pan",
        template="plotly_dark",
        paper_bgcolor="#12171e",
        plot_bgcolor="#12171e",
        font=dict(color="#c7cdd6", family="Inter, sans-serif"),
        title=dict(
            text=f"종료 캔들 전체 {len(chart):,}개 · 초기 화면 최근 100개",
            font=dict(color="#c9a24b", size=14, family="JetBrains Mono, monospace"),
        ),
        legend=dict(
            orientation="h", x=0, y=1.04,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
        uirevision=f"{timestamps.iloc[-1]}-{len(chart)}",
    )
    axis_style = dict(
        gridcolor="rgba(43, 51, 61, 0.48)",
        zeroline=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikedash="dot",
        spikecolor="rgba(202, 210, 219, 0.55)",
        spikethickness=1,
    )
    figure.update_xaxes(
        **axis_style,
        tickformat="%m.%d.%Y",
        showgrid=True,
        rangeslider_visible=False,
        range=[initial_start, zone_end],
    )
    figure.update_yaxes(
        **axis_style,
        title_text="가격 (USDT)",
        side="right",
        nticks=6,
        range=[y_min, y_max],
        tickformat=",.2f",
    )
    return figure


def probability_figure(probability_up: float) -> go.Figure:
    up = float(probability_up) * 100
    down = 100 - up
    figure = go.Figure()
    figure.add_bar(
        y=["확률"], x=[up], orientation="h", name="상승", marker_color="#4fd1ae",
        text=[f"상승 {up:.2f}%"], textposition="inside",
        textfont=dict(color="#07110d", size=11, family="JetBrains Mono, monospace"),
        insidetextanchor="middle",
    )
    figure.add_bar(
        y=["확률"], x=[down], orientation="h", name="하락", marker_color="#ef6461",
        text=[f"하락 {down:.2f}%"], textposition="inside",
        textfont=dict(color="#1c0908", size=11, family="JetBrains Mono, monospace"),
        insidetextanchor="middle",
    )
    figure.update_layout(
        barmode="stack",
        height=105,
        margin=dict(l=4, r=4, t=18, b=4),
        xaxis=dict(
            range=[0, 100], ticksuffix="%", fixedrange=True,
            gridcolor="#1c222a", zerolinecolor="#1c222a"
        ),
        yaxis=dict(visible=False, fixedrange=True),
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="#12171e",
        plot_bgcolor="#12171e",
        font=dict(color="#c7cdd6", family="Inter, sans-serif"),
    )
    return figure


def reliability_status(summary: dict[str, Any]) -> str:
    """Use probability quality, fold stability, and sample size conservatively."""
    mode = str(summary.get("validation_mode", "STANDARD")).upper()
    if mode == "INSUFFICIENT_HISTORY":
        return "데이터 이력 부족 / OOS 판단 불가"
    if mode == "LIMITED_HISTORY":
        return "LIMITED_HISTORY / 판단 주의"
    if mode == "SHORT_HISTORY":
        return "SHORT_HISTORY / 신뢰 제한"

    auc = float(summary.get("roc_auc", 0.5))
    fold_ratio = float(summary.get("folds_auc_above_0_5", 0.0))
    gap = float(summary.get("calibration_gap", 1.0))
    samples = int(summary.get("oos_samples", 0))
    folds = int(summary.get("folds", 0))
    if samples < 500 or folds < 3:
        return "표본 또는 Fold 부족"
    if auc <= 0.50:
        return "무작위 수준 이하"
    if gap > 0.05:
        return "확률 보정 오차 큼 / 관찰 필요"
    if fold_ratio < 0.60:
        return "시장 구간별 성능 불안정"
    if auc < 0.53:
        return "매우 약한 예측력"
    if auc < 0.56:
        return "약한 예측력 / 관찰 필요"
    return "상대적으로 양호 / 지속 검증 필요"


class ValidationProgressWriter(io.StringIO):
    """Capture the existing validator's prints and mirror Fold progress to UI."""

    def __init__(self, progress, status_text, expected_folds: int):
        super().__init__()
        self.progress = progress
        self.status_text = status_text
        self.expected_folds = max(1, int(expected_folds))
        self.pending = ""

    def write(self, text: str) -> int:
        written = super().write(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            match = re.search(r"Fold\s+(\d+)", line)
            if match:
                fold = min(int(match.group(1)), self.expected_folds)
                self.status_text.info(
                    f"Walk-Forward Fold {fold}/{self.expected_folds} 완료"
                )
                self.progress.progress(0.15 + 0.80 * fold / self.expected_folds)
        return written


def run_latest_action(symbol: str, timeframe: str) -> dict[str, Any]:
    log = io.StringIO()
    try:
        with redirect_stdout(log), redirect_stderr(log):
            result = core.run_latest_prediction(
                symbol=symbol,
                timeframe=timeframe,
                refresh=True,
                force_retrain=False,
            )
            # 방향 모델이 이미 데이터를 갱신했으므로 분위수 모델은 로컬 CSV를
            # 재사용한다. 고정 Holdout 결과는 읽기만 하고 다시 선택에 쓰지 않는다.
            range_result = range_live.run_latest_range_prediction(
                symbol=symbol,
                timeframe=timeframe,
                refresh=False,
                force_retrain=False,
            )
            raw_df = core.load_raw_data(symbol, timeframe, refresh=False)
            summary = validation.load_validation_summary(symbol, timeframe)
    except Exception as exc:
        exc.operation_log = log.getvalue()
        raise
    return {
        "result": result,
        "range_result": range_result,
        "raw_df": raw_df,
        "validation_summary": summary,
        "operation_log": log.getvalue(),
        "notice": "최신 예측을 확인했습니다.",
    }


def run_refresh_action(symbol: str, timeframe: str) -> dict[str, Any]:
    log = io.StringIO()
    try:
        with redirect_stdout(log), redirect_stderr(log):
            raw_df = core.load_raw_data(symbol, timeframe, refresh=True)
            summary = validation.load_validation_summary(symbol, timeframe)
    except Exception as exc:
        exc.operation_log = log.getvalue()
        raise
    return {
        "result": None,
        "raw_df": raw_df,
        "validation_summary": summary,
        "operation_log": log.getvalue(),
        "notice": (
            f"종료된 캔들 {len(raw_df):,}개를 확인했습니다. "
            "최신 예측을 보려면 ‘최신 예측 확인’을 누르세요."
        ),
    }


def run_validation_action(
    symbol: str,
    timeframe: str,
    run_noise_test: bool = False,
) -> dict[str, Any]:
    paths = core.get_paths(symbol, timeframe)
    replacing = validation.load_validation_summary(symbol, timeframe) is not None
    if replacing:
        st.info("기존 검증 결과를 새 결과로 갱신합니다.")

    with st.status(
        "Walk-Forward 검증을 준비하고 있습니다.", expanded=True
    ) as status:
        progress = st.progress(0.02)
        status_text = st.empty()
        prepare_log = io.StringIO()
        try:
            status_text.info("저장 데이터와 피처를 확인하는 중입니다.")
            with redirect_stdout(prepare_log), redirect_stderr(prepare_log):
                raw_df, _, model_df = core.prepare_data(
                    symbol, timeframe, refresh=False
                )
            development_df, _, _ = validation.split_development_lockbox(model_df)
            schedule = validation.build_fold_schedule(development_df, timeframe)
            expected_folds = len(schedule)
            if expected_folds == 0:
                # Let the existing validation function produce its authoritative
                # history-shortage message below.
                expected_folds = 1
            progress.progress(0.12)
            status_text.info(
                f"검증 구간 확인 완료 · 예정 Fold {len(schedule)}개"
            )
            writer = ValidationProgressWriter(
                progress, status_text, expected_folds
            )
            with redirect_stdout(writer), redirect_stderr(writer):
                summary = validation.run_walk_forward(
                    symbol=symbol,
                    timeframe=timeframe,
                    refresh=False,
                    run_noise_test=run_noise_test,
                )
            progress.progress(1.0)
            status.update(label="Walk-Forward 검증이 완료되었습니다.", state="complete")
            return {
                "result": None,
                "raw_df": raw_df,
                "validation_summary": summary,
                "operation_log": prepare_log.getvalue() + writer.getvalue(),
                "notice": "OOS 검증 결과를 새로 저장했습니다.",
            }
        except Exception as exc:
            captured = prepare_log.getvalue()
            if "writer" in locals():
                captured += writer.getvalue()
            exc.operation_log = captured
            status.update(label="Walk-Forward 검증을 완료하지 못했습니다.", state="error")
            raise


def run_lockbox_action(symbol: str, timeframe: str) -> dict[str, Any]:
    """예약된 최종 90일 미공개 검증 데이터를 한 번만 평가한다."""
    log = io.StringIO()
    try:
        with st.status("최종 미공개 데이터를 검증하고 있습니다.", expanded=True) as status:
            with redirect_stdout(log), redirect_stderr(log):
                lockbox = validation.evaluate_lockbox_once(
                    symbol=symbol,
                    timeframe=timeframe,
                    refresh=False,
                    force=False,
                )
                raw_df = core.load_raw_data(symbol, timeframe, refresh=False)
                summary = validation.load_validation_summary(symbol, timeframe)
            status.update(label="최종 미공개 데이터 검증이 완료되었습니다.", state="complete")
    except Exception as exc:
        exc.operation_log = log.getvalue()
        raise
    return {
        "result": None,
        "raw_df": raw_df,
        "validation_summary": summary,
        "operation_log": log.getvalue(),
        "notice": (
            "최종 미공개 데이터 검증 결과를 확인했습니다. 이 구간을 본 뒤에는 같은 결과로 "
            "피처·하이퍼파라미터·임계값을 다시 선택하면 안 됩니다."
        ),
        "lockbox_result": lockbox,
    }


def render_recent_data(
    raw_df: pd.DataFrame,
    prediction: dict[str, Any] | None = None,
) -> None:
    st.plotly_chart(
        candlestick_figure(raw_df, prediction),
        use_container_width=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "doubleClick": "reset+autosize",
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        },
    )
    st.caption(
        "마우스 휠: 확대·축소 · 드래그: 좌우 이동 · 더블클릭: 전체 범위 초기화 · "
        "우측 상단 Autoscale: 현재 선택 범위에 맞춤"
    )


def render_prediction(result: dict[str, Any]) -> None:
    token = html.escape(result["symbol"].split("/")[0])
    timeframe_label = TIMEFRAME_LABELS[result["timeframe"]]
    st.subheader(f"{token} {timeframe_label} 예측")

    signal = str(result["signal"]).upper()
    signal_class = {
        "LONG": "ticker-long",
        "SHORT": "ticker-short",
        "NEUTRAL": "ticker-neutral",
    }.get(signal, "ticker-neutral")
    glyph = {"LONG": "▲", "SHORT": "▼", "NEUTRAL": "–"}.get(signal, "–")
    st.markdown(
        f'<div class="ticker-strip {signal_class}">'
        f'<span class="ticker-glyph">{glyph}</span>'
        f'<span class="ticker-symbol">{token}·{html.escape(timeframe_label)}</span>'
        f'<span class="ticker-sep">|</span>'
        f'<span class="ticker-signal">{html.escape(signal)}</span>'
        f'<span class="ticker-sep">|</span>'
        f'<span class="ticker-prob">↑{result["probability_up"]:.1%} '
        f'↓{result["probability_down"]:.1%}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    first = st.columns(4)
    first[0].metric("기준 종가 (USDT)", f"{result['reference_close']:,.2f}")
    first[1].metric("상승 확률", f"{result['probability_up']:.2%}")
    first[2].metric("하락 확률", f"{result['probability_down']:.2%}")
    first[3].metric("신호 기준 확률", f"{result['threshold']:.2%}")
    second = st.columns(3)
    second[0].metric("확률 보정", str(result["calibration_method"]))
    second[1].metric("모델 상태", "재사용" if result["model_reused"] else "새로 학습")
    second[2].metric(
        "예측 기록",
        "신규 저장" if result["prediction_created"] else "기존 기록 재사용",
    )

    st.plotly_chart(
        probability_figure(result["probability_up"]),
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )
    st.markdown(
        f"**기준 캔들:** {format_timestamp(result['source_candle_start'])} ~ "
        f"{format_timestamp(result['source_candle_end'])}  \n"
        f"**예측 대상:** {format_timestamp(result['target_candle_start'])} ~ "
        f"{format_timestamp(result['target_candle_end'])}  \n"
        f"**모델 학습 시각:** {format_timestamp(result['trained_at'])}"
    )
    if signal == "NEUTRAL":
        st.caption("상승·하락 어느 쪽도 거래 임계값에 도달하지 않아 NEUTRAL입니다.")


def direction_gauge_figure(result: dict[str, Any]) -> go.Figure:
    """분류확률을 0~100 방향 눈금으로 보여준다."""
    probability_up = float(result["probability_up"]) * 100.0
    signal = str(result["signal"]).upper()
    signal_color = {
        "LONG": "#4fd1ae",
        "SHORT": "#ef6461",
        "NEUTRAL": "#c9a24b",
    }.get(signal, "#c9a24b")
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_up,
            number={"suffix": "%", "font": {"size": 34, "color": signal_color}},
            title={"text": signal, "font": {"size": 14, "color": signal_color}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 50, 100],
                    "ticktext": ["하락", "중립", "상승"],
                    "tickfont": {"size": 10, "color": "#8892a0"},
                },
                "bar": {"color": signal_color, "thickness": 0.16},
                "bgcolor": "#0d1116",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45], "color": "rgba(239,100,97,.72)"},
                    {"range": [45, 55], "color": "rgba(201,162,75,.72)"},
                    {"range": [55, 100], "color": "rgba(79,209,174,.72)"},
                ],
                "threshold": {
                    "line": {"color": "#f4f6f8", "width": 3},
                    "thickness": 0.78,
                    "value": probability_up,
                },
            },
        )
    )
    figure.update_layout(
        height=270,
        margin=dict(l=24, r=24, t=36, b=10),
        paper_bgcolor="#0d1116",
        plot_bgcolor="#0d1116",
        font=dict(family="Inter, sans-serif", color="#e8ecf1"),
    )
    return figure


def render_candle_insight(
    direction_result: dict[str, Any],
    range_result: dict[str, Any] | None,
) -> None:
    """방향 확률과 별도 분위수 회귀의 예상 가격구간을 함께 표시한다."""
    token = html.escape(direction_result["symbol"].split("/")[0])
    timeframe = direction_result["timeframe"]
    timeframe_label = html.escape(TIMEFRAME_LABELS[timeframe])
    st.markdown(
        f'<div class="insight-title-row">'
        f'<div><span class="insight-bulb">◉</span>'
        f'<span class="insight-title">캔들 인사이트</span></div>'
        f'<div><span class="insight-badge">{token}USDT</span>'
        f'<span class="insight-badge">{html.escape(timeframe.upper())}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if range_result is None:
        st.warning(
            "예상 가격구간을 아직 계산하지 못했습니다. 최신 예측을 다시 실행하면 "
            "Q10·Q50·Q90 운영 모델을 준비합니다."
        )
        return

    left, right = st.columns([0.38, 0.62], gap="large")
    with left:
        st.markdown('<div class="insight-kicker">방향 확률 지수</div>', unsafe_allow_html=True)
        st.plotly_chart(
            direction_gauge_figure(direction_result),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )
        signal = html.escape(str(direction_result["signal"]).upper())
        st.markdown(
            f'<div class="insight-note">다음 {timeframe_label} 캔들에 대해 '
            f'<strong>{signal}</strong> 신호입니다. 상승확률 '
            f'<strong>{direction_result["probability_up"]:.2%}</strong>, 하락확률 '
            f'<strong>{direction_result["probability_down"]:.2%}</strong>이며 '
            f'신호 기준 확률은 {direction_result["threshold"]:.2%}입니다.</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="insight-kicker">다음 1봉 예상 수익률</div>', unsafe_allow_html=True)
        returns = st.columns(3)
        returns[0].metric("하단 Q10", f"{range_result['q10_return']:+.2%}")
        returns[1].metric("중앙 Q50", f"{range_result['q50_return']:+.2%}")
        returns[2].metric("상단 Q90", f"{range_result['q90_return']:+.2%}")

        rows = [
            ("예상 하단", "Q10", range_result["q10_price"], range_result["q10_return"]),
            ("예상 중앙", "Q50", range_result["q50_price"], range_result["q50_return"]),
            ("예상 상단", "Q90", range_result["q90_price"], range_result["q90_return"]),
        ]
        body = "".join(
            '<div class="range-row">'
            f'<div><b>{html.escape(label)}</b><span>{quantile}</span></div>'
            f'<div class="range-price">{price:,.2f}<small> USDT</small></div>'
            f'<div class="range-return {"range-up" if change >= 0 else "range-down"}">'
            f'{change:+.2%}</div></div>'
            for label, quantile, price, change in rows
        )
        st.markdown(
            '<div class="range-table">'
            '<div class="range-head"><span>예상 구간</span><span>예상 가격</span><span>기준 종가 대비</span></div>'
            f'{body}</div>',
            unsafe_allow_html=True,
        )

        quality = range_result.get("quality_summary") or {}
        if quality:
            evaluation = str(quality.get("evaluation", "OOS"))
            st.markdown(
                '<div class="range-quality">'
                f'<span>Q10~Q90 포함률 <b>{float(quality.get("interval_coverage", 0)):.2%}</b></span>'
                f'<span>Pinball Skill <b>{float(quality.get("pinball_skill_score", 0)):+.2%}</b></span>'
                f'<span>{html.escape(evaluation)}</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("저장된 분위수 OOS 품질 결과가 없어 예상값만 표시합니다.")

    st.caption(
        "예상 하단·중앙·상단은 목표가나 손절가가 아니라 조건부 Q10·Q50·Q90입니다. "
        "Q10~Q90 밖의 가격도 발생할 수 있습니다."
    )


def _read_csv_if_available(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()
    return frame


def _render_diagnostic_charts(symbol: str, timeframe: str) -> None:
    paths = core.get_paths(symbol, timeframe, create=False)
    folds = _read_csv_if_available(paths.validation_folds)
    predictions = _read_csv_if_available(paths.validation_predictions)
    reliability = _read_csv_if_available(paths.validation_reliability)
    importance = _read_csv_if_available(paths.validation_permutation)

    with st.expander("진단 그래프와 변수 중요도", expanded=False):
        raw_columns = {
            "fold", "train_raw_roc_auc", "validation_raw_roc_auc", "oos_raw_roc_auc"
        }
        if not folds.empty and raw_columns.issubset(folds.columns):
            figure = go.Figure()
            for column, label, color in (
                ("train_raw_roc_auc", "Train", "#c9a24b"),
                ("validation_raw_roc_auc", "Validation", "#7b8da6"),
                ("oos_raw_roc_auc", "OOS", "#4fd1ae"),
            ):
                figure.add_trace(go.Scatter(
                    x=folds["fold"], y=folds[column], mode="lines+markers",
                    name=label, line={"color": color, "width": 2},
                ))
            figure.add_hline(y=0.5, line_dash="dot", line_color="#5b6470")
            figure.update_layout(
                title="Fold별 Raw Train / Validation / OOS ROC-AUC",
                template="plotly_dark", height=330, margin=dict(l=25, r=20, t=55, b=30),
                paper_bgcolor="#0d1116", plot_bgcolor="#12171e",
                xaxis_title="Fold", yaxis_title="ROC-AUC", hovermode="x unified",
            )
            st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})

        chart_columns = st.columns(2)
        if not reliability.empty and {
            "mean_probability", "actual_up_ratio"
        }.issubset(reliability.columns):
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="이상적 보정",
                line={"color": "#5b6470", "dash": "dot"},
            ))
            figure.add_trace(go.Scatter(
                x=reliability["mean_probability"], y=reliability["actual_up_ratio"],
                mode="lines+markers", name="Model A", line={"color": "#4fd1ae"},
            ))
            figure.update_layout(
                title="Calibration reliability", template="plotly_dark", height=310,
                margin=dict(l=25, r=15, t=55, b=30), paper_bgcolor="#0d1116",
                plot_bgcolor="#12171e", xaxis_title="예측 상승확률",
                yaxis_title="실제 상승비율", xaxis_range=[0, 1], yaxis_range=[0, 1],
            )
            chart_columns[0].plotly_chart(
                figure, use_container_width=True, config={"displaylogo": False}
            )

        if not predictions.empty and {"equity", "drawdown"}.issubset(predictions.columns):
            x = pd.to_datetime(predictions.get("timestamp"), errors="coerce", utc=True)
            figure = make_subplots(specs=[[{"secondary_y": True}]])
            figure.add_trace(go.Scatter(
                x=x, y=predictions["equity"], name="자산곡선",
                line={"color": "#4fd1ae", "width": 2},
            ), secondary_y=False)
            figure.add_trace(go.Scatter(
                x=x, y=predictions["drawdown"], name="Drawdown",
                fill="tozeroy", line={"color": "#ef6461", "width": 1},
                opacity=0.45,
            ), secondary_y=True)
            figure.update_layout(
                title="OOS 누적 자산과 Drawdown", template="plotly_dark", height=310,
                margin=dict(l=25, r=15, t=55, b=30), paper_bgcolor="#0d1116",
                plot_bgcolor="#12171e", hovermode="x unified",
            )
            chart_columns[1].plotly_chart(
                figure, use_container_width=True, config={"displaylogo": False}
            )

        if not importance.empty and {
            "feature", "importance_auc_drop"
        }.issubset(importance.columns):
            ranked = (
                importance.groupby("feature", as_index=False)["importance_auc_drop"]
                .mean().sort_values("importance_auc_drop").tail(12)
            )
            figure = go.Figure(go.Bar(
                x=ranked["importance_auc_drop"], y=ranked["feature"],
                orientation="h", marker_color="#c9a24b",
            ))
            figure.update_layout(
                title="OOS permutation importance · 평균 AUC 감소",
                template="plotly_dark", height=390, margin=dict(l=25, r=20, t=55, b=30),
                paper_bgcolor="#0d1116", plot_bgcolor="#12171e",
                xaxis_title="피처를 섞었을 때 AUC 감소량",
            )
            st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})
            st.caption("값이 클수록 해당 피처를 섞었을 때 OOS 구분력이 더 많이 감소했습니다.")


def _render_baselines(summary: dict[str, Any]) -> None:
    baselines = summary.get("baselines") or {}
    if not baselines:
        return
    labels = {
        "xgboost_model_a": "XGBoost Model A",
        "historical_rate": "과거 상승비율 고정",
        "simple_momentum": "단순 모멘텀",
        "logistic_regression": "로지스틱 회귀",
    }
    rows = []
    for key, metrics in baselines.items():
        rows.append({
            "모델": labels.get(key, key),
            "ROC-AUC": metrics.get("roc_auc"),
            "Log Loss": metrics.get("log_loss"),
            "Brier": metrics.get("brier"),
        })
    with st.expander("기준 모델 비교", expanded=False):
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("모든 기준 모델도 각 Fold의 과거 구간만 사용해 예측했습니다.")


def _render_lockbox(summary: dict[str, Any]) -> None:
    lockbox = summary.get("lockbox") or {}
    st.subheader("최종 미공개 데이터 검증")
    status = str(lockbox.get("status", "RESERVED_NOT_EVALUATED"))
    if status != "EVALUATED_ONCE":
        st.info(
            "가장 최근 90일은 검증용 Holdout으로 분리되어 피처·하이퍼파라미터·"
            "보정법·임계값 선택에서 제외됩니다. 모든 연구 결정을 확정한 뒤에만 "
            "한 번 검증하세요."
        )
        return
    row = st.columns(4)
    row[0].metric("Holdout ROC-AUC", metric_text(lockbox.get("roc_auc"), lambda x: f"{x:.4f}"))
    row[1].metric("Holdout Log Loss", metric_text(lockbox.get("log_loss"), lambda x: f"{x:.4f}"))
    row[2].metric("Holdout Brier", metric_text(lockbox.get("brier"), lambda x: f"{x:.4f}"))
    row[3].metric("Holdout 표본", f"{int(lockbox.get('lockbox_samples', 0)):,}개")
    st.caption(
        f"평가 범위: {format_timestamp(lockbox.get('lockbox_start'), False)} ~ "
        f"{format_timestamp(lockbox.get('lockbox_end'), False)} · "
        f"평가 시각: {format_timestamp(lockbox.get('evaluated_at'), False)}"
    )
    st.warning(
        "최종 미공개 데이터 결과를 본 뒤 같은 구간을 이용해 모델을 다시 선택하면 "
        "검증용 Holdout의 의미가 사라집니다."
    )


def _render_reproducibility(summary: dict[str, Any]) -> None:
    info = summary.get("reproducibility") or {}
    if not info:
        return
    with st.expander("재현성 정보", expanded=False):
        st.json(info, expanded=False)


def render_validation_summary(
    summary: dict[str, Any] | None,
    symbol: str,
    timeframe: str,
) -> None:
    st.subheader("모델 OOS 신뢰도")
    if summary is None:
        st.warning(
            "저장된 OOS 검증 결과가 없습니다.\n\n"
            "사이드바의 ‘개발구간 백테스트·검증’ 버튼을 눌러 검증을 실행하세요."
        )
        return

    folds = int(summary.get("folds", 0))
    fold_ratio = float(summary.get("folds_auc_above_0_5", 0.0))
    folds_above = int(round(folds * fold_ratio))
    mode = str(summary.get("validation_mode", "STANDARD"))
    generated = format_timestamp(summary.get("generated_at"), include_timezone=False)

    row1 = st.columns(4)
    row1[0].metric("ROC-AUC", metric_text(summary.get("roc_auc"), lambda x: f"{x:.4f}"))
    row1[1].metric("Log Loss", metric_text(summary.get("log_loss"), lambda x: f"{x:.4f}"))
    row1[2].metric("Brier Score", metric_text(summary.get("brier"), lambda x: f"{x:.4f}"))
    row1[3].metric(
        "Calibration gap",
        metric_text(summary.get("calibration_gap"), lambda x: f"{x:.4f}"),
    )
    row2 = st.columns(4)
    row2[0].metric(
        "Fold 평균 AUC",
        metric_text(summary.get("fold_mean_auc"), lambda x: f"{x:.4f}"),
    )
    row2[1].metric("AUC>0.5 Fold", f"{folds_above}/{folds} ({fold_ratio:.2%})")
    row2[2].metric("OOS 표본", f"{int(summary.get('oos_samples', 0)):,}개")
    row2[3].metric("Fold 수", f"{folds:,}개")

    row3 = st.columns(4)
    row3[0].metric(
        "Train 평균 AUC",
        metric_text(summary.get("train_mean_auc"), lambda x: f"{x:.4f}"),
    )
    row3[1].metric(
        "Validation 평균 AUC",
        metric_text(summary.get("validation_mean_auc"), lambda x: f"{x:.4f}"),
    )
    row3[2].metric(
        "Raw Train–OOS gap",
        metric_text(summary.get("generalization_gap"), lambda x: f"{x:.4f}"),
    )
    row3[3].metric(
        "Fold AUC 표준편차",
        metric_text(summary.get("fold_auc_std"), lambda x: f"{x:.4f}"),
    )

    confidence = summary.get("auc_confidence_interval") or {}
    if confidence.get("status") == "AVAILABLE":
        st.caption(
            "Block bootstrap OOS AUC 95% 신뢰구간: "
            f"{float(confidence['auc_ci_95_low']):.4f} ~ "
            f"{float(confidence['auc_ci_95_high']):.4f} · "
            f"최적 트리 수 평균 {float(summary.get('best_trees_mean', 0)):.1f}개 "
            f"({int(summary.get('best_trees_min', 0))}~{int(summary.get('best_trees_max', 0))})"
        )

    state = reliability_status(summary)
    st.markdown(
        f"**검증 모드:** `{mode}`  \n"
        f"**검증 시각:** {generated}  \n"
        f"**검증 상태:** {state}  \n"
        f"**과적합 진단:** {summary.get('overfit_status', '계산 불가')}"
    )
    if mode in {"SHORT_HISTORY", "LIMITED_HISTORY"}:
        st.warning("단기 이력 검증은 STANDARD 검증과 같은 신뢰도로 비교할 수 없습니다.")

    _render_baselines(summary)
    _render_diagnostic_charts(symbol, timeframe)
    noise = summary.get("noise_test") or {}
    if noise.get("status") == "COMPLETED":
        st.caption(
            "반복 노이즈 피처 상위 20% 진입률: "
            f"{float(noise.get('top_20pct_rate', 0)):.2%}"
        )
    else:
        st.caption("노이즈 피처 반복 실험은 이번 검증에서 실행하지 않았습니다.")
    render_strategy(summary.get("strategy", {}))
    _render_lockbox(summary)
    _render_reproducibility(summary)


def render_strategy(strategy: dict[str, Any]) -> None:
    st.subheader("비용 반영 전략 성과")
    row1 = st.columns(4)
    row1[0].metric("종료 거래", f"{int(strategy.get('closed_trades', 0)):,}회")
    row1[1].metric("승률", metric_text(strategy.get("win_rate"), lambda x: f"{x:.2%}"))
    row1[2].metric(
        "평균 수익률", metric_text(strategy.get("average_win"), lambda x: f"{x:.2%}")
    )
    row1[3].metric(
        "평균 손실률", metric_text(strategy.get("average_loss"), lambda x: f"{x:.2%}")
    )
    row2 = st.columns(4)
    row2[0].metric(
        "손익비", metric_text(strategy.get("payoff_ratio"), lambda x: f"{x:.2f}")
    )
    row2[1].metric(
        "거래당 기대값", metric_text(strategy.get("expectancy"), lambda x: f"{x:.2%}")
    )
    row2[2].metric(
        "Profit Factor", metric_text(strategy.get("profit_factor"), lambda x: f"{x:.2f}")
    )
    row2[3].metric(
        "최다 연속 손실", f"{int(strategy.get('max_consecutive_losses', 0)):,}회"
    )
    row3 = st.columns(4)
    row3[0].metric(
        "누적수익률",
        metric_text(strategy.get("cumulative_return"), lambda x: f"{x:.2%}"),
    )
    row3[1].metric("Sharpe", metric_text(strategy.get("sharpe"), lambda x: f"{x:.2f}"))
    row3[2].metric("Sortino", metric_text(strategy.get("sortino"), lambda x: f"{x:.2f}"))
    row3[3].metric("MDD", metric_text(strategy.get("mdd"), lambda x: f"{x:.2%}"))
    st.metric(
        "총수수료 (OOS 기간 자본 대비 합계)",
        metric_text(strategy.get("total_fee"), lambda x: f"{x:.2%}"),
    )
    st.warning("과거 OOS 모의검증 결과이며 미래 수익을 보장하지 않습니다.")


def render_model_info(
    symbol: str,
    timeframe: str,
    raw_df: pd.DataFrame,
    result: dict[str, Any] | None,
) -> None:
    paths = core.get_paths(symbol, timeframe, create=False)
    metadata = load_json_file(paths.metadata)
    with st.expander("데이터 및 모델 정보"):
        first = format_timestamp(raw_df["timestamp"].min(), include_timezone=False)
        last = format_timestamp(raw_df["timestamp"].max(), include_timezone=False)
        st.write(f"데이터 범위: {first} ~ {last}")
        st.write(f"종료 캔들 수: {len(raw_df):,}개")
        st.write(f"모델: {core.MODEL_NAME} · {core.MODEL_VERSION}")
        st.write(f"입력 피처: {len(core.FEATURE_COLUMNS)}개")
        if result is not None:
            st.write(f"모델 학습 시각: {format_timestamp(result['trained_at'])}")
        if metadata.get("available_labeled_samples") is not None:
            st.write(f"학습 가능 표본: {int(metadata['available_labeled_samples']):,}개")
        st.write(f"데이터 저장 위치: {paths.root}")

        effective = core.LEVERAGE * core.CAPITAL_FRACTION
        one_way_equity_fee = effective * core.ONE_WAY_FEE_RATE
        round_trip_equity_fee = 2 * one_way_equity_fee
        st.markdown(
            "**현재 비용 가정**  \n"
            f"레버리지 {core.LEVERAGE:.1f}배 · 자본 투입 {core.CAPITAL_FRACTION:.0%} · "
            f"기초자산 편도 수수료 {core.ONE_WAY_FEE_RATE:.3%} · "
            f"자본 대비 편도 {one_way_equity_fee:.3%} · "
            f"왕복 {round_trip_equity_fee:.3%} · "
            f"슬리피지 {core.SLIPPAGE_RATE:.3%} · 펀딩비 제외"
        )


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

        :root {
            --bg: #0a0d11;
            --surface: #0d1116;
            --card: #12171e;
            --border: #212830;
            --border-strong: #323c48;
            --text: #e8ecf1;
            --muted: #8892a0;
            --muted-dim: #5b6470;
            --gold: #c9a24b;
            --gold-soft: #8a7238;
            --teal: #4fd1ae;
            --coral: #ef6461;
            --font-display: "Space Grotesk", "Pretendard", sans-serif;
            --font-body: "Inter", "Pretendard", -apple-system, sans-serif;
            --font-mono: "JetBrains Mono", "SFMono-Regular", monospace;
        }

        html, body, [class*="css"] {font-family: var(--font-body); font-size: 14px;}
        .stApp, [data-testid="stAppViewContainer"] {background: var(--bg); color: var(--text);}
        [data-testid="stHeader"] {background: rgba(10,13,17,.94); border-bottom: 1px solid var(--border);}
        /*
         * Deploy와 우측 메뉴만 숨긴다.
         * stToolbar 전체를 숨기면 사이드바를 접은 뒤 다시 여는
         * stSidebarCollapsedControl까지 함께 사라지므로 숨기면 안 된다.
         */
        [data-testid="stAppDeployButton"],
        #MainMenu {display: none !important; visibility: hidden !important;}

        /* 사이드바가 접힌 상태에서도 '설정 열기' 버튼을 항상 노출한다. */
        [data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            position: fixed !important;
            top: .65rem !important;
            left: .65rem !important;
            z-index: 100000 !important;
        }
        [data-testid="stSidebarCollapsedControl"] button {
            color: var(--text) !important;
            background: var(--card) !important;
            border: 1px solid var(--border-strong) !important;
            border-radius: 4px !important;
        }
        [data-testid="stSidebarCollapsedControl"] button:hover {
            color: var(--gold) !important;
            border-color: var(--gold) !important;
        }
        [data-testid="stSidebar"] {background: var(--surface); border-right: 1px solid var(--border);}
        [data-testid="stSidebar"] hr {border-color: var(--border);}
        [data-testid="stSidebar"] h2 {
            font-family: var(--font-display); font-weight: 700;
            font-size: 1rem !important; letter-spacing: -.01em;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {color: var(--text);}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: var(--muted-dim);}

        .block-container {max-width: 1160px; padding-top: 2.6rem; padding-bottom: 4rem;}

        /* ---- masthead: flush, no card-in-a-card ---- */
        .masthead {
            border-top: 2px solid var(--gold);
            padding: 1rem 0 1.25rem; margin-bottom: 0;
        }
        .masthead-row {
            display: flex; align-items: baseline; justify-content: space-between;
            flex-wrap: wrap; gap: .4rem .9rem; margin-bottom: .7rem;
        }
        .wordmark {
            font-family: var(--font-mono); font-weight: 700; font-size: .82rem;
            letter-spacing: .12em; color: var(--text);
        }
        .wordmark-mark {color: var(--gold); padding-right: .3rem;}
        .masthead-tag {
            font-family: var(--font-mono); font-size: .68rem; letter-spacing: .1em;
            color: var(--muted-dim);
        }
        .masthead h1 {
            font-family: var(--font-display) !important; font-weight: 700 !important;
            color: var(--text) !important; font-size: clamp(1.5rem, 3vw, 2rem) !important;
            margin: 0 0 .5rem !important; letter-spacing: -.02em !important; line-height: 1.15 !important;
        }
        .masthead p {
            margin: 0; color: var(--muted); font-size: .88rem;
            line-height: 1.6; max-width: 700px;
        }

        /* ---- section headers: hairline rule, not a colored block ---- */
        h2 {
            font-family: var(--font-display) !important; font-weight: 700 !important;
            color: var(--text) !important; letter-spacing: -.01em !important;
            font-size: 1.08rem !important;
            margin-top: 2rem !important; padding-bottom: .55rem !important;
            border-bottom: 1px solid var(--border) !important;
        }
        h3 {font-family: var(--font-display); color: var(--text);}
        p, span, label {color: var(--text);}

        /* ---- ticker strip: the signature element ---- */
        .ticker-strip {
            display: flex; align-items: center; gap: .65rem; flex-wrap: wrap;
            font-family: var(--font-mono); font-size: .86rem; letter-spacing: .02em;
            padding: .7rem 1rem; margin: .35rem 0 1rem 0;
            background: var(--card); border: 1px solid var(--border);
            border-left: 3px solid var(--muted-dim); border-radius: 3px;
        }
        .ticker-glyph {font-size: 1rem; font-weight: 700;}
        .ticker-symbol {color: var(--muted); font-weight: 500;}
        .ticker-sep {color: var(--border-strong);}
        .ticker-signal {font-weight: 700; letter-spacing: .06em;}
        .ticker-prob {color: var(--muted); margin-left: auto;}
        .ticker-long {border-left-color: var(--teal);}
        .ticker-long .ticker-glyph, .ticker-long .ticker-signal {color: var(--teal);}
        .ticker-short {border-left-color: var(--coral);}
        .ticker-short .ticker-glyph, .ticker-short .ticker-signal {color: var(--coral);}
        .ticker-neutral {border-left-color: var(--gold);}
        .ticker-neutral .ticker-glyph, .ticker-neutral .ticker-signal {color: var(--gold);}

        /* ---- metrics: ledger rows, not boxed cards ---- */
        [data-testid="stMetric"] {
            background: transparent; border: none; border-top: 1px solid var(--border);
            border-radius: 0; padding: .55rem 0 .1rem; min-height: unset;
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted); font-size: .72rem; letter-spacing: .04em;
            text-transform: uppercase;
        }
        [data-testid="stMetricValue"] {
            color: var(--text); font-weight: 500; font-family: var(--font-mono);
            font-size: clamp(1.05rem, 1.5vw, 1.55rem); line-height: 1.3;
        }
        [data-testid="stMetricValue"] > div {
            overflow: visible !important; text-overflow: clip !important; white-space: nowrap !important;
        }

        [data-testid="stDataFrame"] {border: 1px solid var(--border); border-radius: 4px; overflow: hidden;}
        [data-testid="stExpander"] {background: var(--card); border: 1px solid var(--border); border-radius: 6px;}
        [data-testid="stExpander"] summary {font-family: var(--font-mono); font-size: .82rem;}

        .stTextInput input {
            color: var(--text) !important; background: var(--card) !important;
            border-color: var(--border) !important; font-family: var(--font-mono) !important;
        }
        [data-baseweb="select"] > div {
            color: var(--text) !important; background: var(--card) !important;
            border-color: var(--border) !important;
        }

        .stButton > button {
            width: 100%; border-radius: 4px; min-height: 2.5rem;
            font-weight: 600; font-family: var(--font-body);
            border: 1px solid var(--border-strong); color: var(--text); background: var(--card);
            transition: border-color .12s ease, color .12s ease;
        }
        .stButton > button:hover {border-color: var(--gold); color: var(--gold); background: var(--surface);}
        .stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
            color: #14100a; background: var(--gold); border-color: var(--gold); font-weight: 700;
        }
        .stButton > button[kind="primary"] *, button[data-testid="stBaseButton-primary"] * {
            color: #14100a !important;
        }
        .stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
            background: #ddb15c; border-color: #ddb15c;
        }

        [data-testid="stAlert"] {
            background: var(--card); border: 1px solid var(--border); border-radius: 4px; color: var(--text);
        }
        code {color: var(--text) !important; background: var(--surface) !important; font-family: var(--font-mono) !important;}
        a {color: var(--gold) !important;}

        .footer-notice {
            margin-top: 1rem; padding: .6rem .8rem; border-radius: 3px;
            background: var(--card); border: 1px solid var(--border); border-left: 2px solid var(--gold);
            color: var(--muted); font-size: .8rem; font-family: var(--font-mono);
        }

        /* ---- candle insight: direction + honest quantile range ---- */
        .insight-title-row {
            display: flex; align-items: center; justify-content: space-between;
            gap: 1rem; margin: 2.2rem 0 1rem; padding: .9rem 1rem;
            background: #111118; border: 1px solid #252630; border-radius: 10px;
        }
        .insight-title {font-family: var(--font-display); font-size: 1.25rem; font-weight: 700;}
        .insight-bulb {color: #f0d765; margin-right: .65rem; font-size: 1.1rem;}
        .insight-badge {
            display: inline-block; margin-left: .4rem; padding: .28rem .55rem;
            border-radius: 5px; background: #292932; color: #aaaab5;
            font-family: var(--font-mono); font-size: .72rem; font-weight: 700;
        }
        .insight-kicker {
            margin: .4rem 0 .7rem; color: var(--text); font-family: var(--font-display);
            font-size: .95rem; font-weight: 700;
        }
        .insight-note {
            min-height: 94px; padding: .9rem 1rem; border-radius: 7px;
            background: #15161d; border: 1px solid #252630;
            color: #c7c9d0; font-size: .82rem; line-height: 1.65;
        }
        .insight-note strong {color: var(--teal);}
        .range-table {margin-top: 1rem; border: 1px solid #252630; border-radius: 7px; overflow: hidden;}
        .range-head, .range-row {
            display: grid; grid-template-columns: 1fr 1.35fr .9fr;
            align-items: center; column-gap: .7rem; padding: .72rem .85rem;
        }
        .range-head {
            background: #0d0e13; color: #858792; font-size: .7rem;
            letter-spacing: .04em; text-transform: uppercase;
        }
        .range-row {background: #17181f; border-top: 1px solid #252630;}
        .range-row:nth-child(odd) {background: #111218;}
        .range-row b {font-size: .82rem; font-weight: 600;}
        .range-row span {display: block; color: #777a86; font-size: .67rem; margin-top: .12rem;}
        .range-price {font-family: var(--font-mono); text-align: right; font-size: .92rem; font-weight: 700;}
        .range-price small {color: #777a86; font-size: .62rem;}
        .range-return {font-family: var(--font-mono); text-align: right; font-size: .82rem; font-weight: 700;}
        .range-up {color: var(--teal);}
        .range-down {color: var(--coral);}
        .range-quality {
            display: flex; flex-wrap: wrap; gap: .5rem 1.2rem; margin-top: .75rem;
            padding: .65rem .8rem; background: #10141a; border-left: 2px solid var(--gold);
            color: #8f98a5; font-family: var(--font-mono); font-size: .7rem;
        }
        .range-quality b {color: #d8dde4;}

        @media (max-width: 640px) {
            .block-container {padding: .8rem .7rem 2rem .7rem;}
            .masthead {padding: .8rem 0 1rem;}
            .ticker-strip {font-size: .76rem; padding: .6rem .75rem;}
            .ticker-prob {margin-left: 0; width: 100%;}
            [data-testid="stMetricValue"] {font-size: 1.15rem;}
            .insight-title-row {align-items: flex-start;}
            .range-head, .range-row {grid-template-columns: .9fr 1.2fr .8fr; padding: .6rem;}
            .range-price {font-size: .78rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="암호화폐 방향 예측 시스템",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()
    st.markdown(
        """
        <div class="masthead">
          <div class="masthead-row">
            <div class="wordmark"><span class="wordmark-mark">/</span>SIGNAL LEDGER</div>
            <div class="masthead-tag">MODEL A · RESEARCH BUILD</div>
          </div>
          <h1>암호화폐 방향 예측 시스템</h1>
          <p>종료된 OHLCV 캔들로 다음 캔들의 상승·하락 확률을 추정합니다.
          저장 모델과 OOS 검증 결과를 한 화면에서 확인할 수 있으며, 실제 주문은 전송되지 않습니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("예측 설정")
        token_input = st.text_input(
            "코인 토큰명",
            value="BTC",
            placeholder="BTC, ETH, XRP, DOGE, SNDK",
            help="Binance USD-M에 상장된 USDT 무기한 선물의 토큰명을 입력하세요.",
        ).strip().upper()
        timeframe = st.selectbox(
            "시간봉",
            options=list(TIMEFRAME_LABELS),
            format_func=lambda value: TIMEFRAME_LABELS[value],
            index=2,
        )
        latest_clicked = st.button(
            "최신 예측 확인", type="primary", use_container_width=True
        )
        refresh_clicked = st.button("데이터 새로고침", use_container_width=True)

        st.divider()
        with st.expander("고급 연구·검증", expanded=False):
            st.caption(
                "모델을 바꾸거나 포트폴리오 검증 자료를 만들 때만 사용합니다. "
                "일상적인 최신 예측에는 필요하지 않습니다."
            )
            run_noise_test = st.checkbox(
                "과적합 검사",
                value=False,
                help="Fold마다 난수 피처를 반복 학습하므로 검증 시간이 더 길어집니다.",
            )
            validation_clicked = st.button(
                "개발구간 백테스트·검증", use_container_width=True
            )
            lockbox_confirmed = st.checkbox(
                "모델과 연구 기준을 모두 확정했습니다",
                value=False,
                help="체크 후 최종 90일을 한 번 평가할 수 있습니다.",
            )
            lockbox_clicked = st.button(
                "최종 미공개 데이터 검증",
                use_container_width=True,
                disabled=not lockbox_confirmed,
            )
        st.caption(
            "페이지를 여는 것만으로 수집·학습·검증은 실행되지 않습니다."
        )

    try:
        _, symbol, timeframe = normalize_selection(token_input, timeframe)
    except Exception as exc:
        show_error(exc)
        return

    if SESSION_RESULTS_KEY not in st.session_state:
        st.session_state[SESSION_RESULTS_KEY] = {}
    result_store: dict[str, dict[str, Any]] = st.session_state[SESSION_RESULTS_KEY]
    key = selection_key(symbol, timeframe)

    try:
        if latest_clicked:
            with st.spinner(
                "최신 종료 캔들을 확인하고 저장된 모델로 예측하는 중입니다. "
                "재학습 시 조금 더 오래 걸릴 수 있습니다."
            ):
                result_store[key] = run_latest_action(symbol, timeframe)
        elif refresh_clicked:
            with st.spinner("Binance에서 신규 종료 캔들을 확인하는 중입니다."):
                result_store[key] = run_refresh_action(symbol, timeframe)
        elif validation_clicked:
            previous_entry = result_store.get(key, {})
            validated_entry = run_validation_action(
                symbol, timeframe, run_noise_test=run_noise_test
            )
            # 검증만 다시 실행해도 직전에 보던 최신 예측 카드는 유지한다.
            if previous_entry.get("result") is not None:
                validated_entry["result"] = previous_entry["result"]
            if previous_entry.get("range_result") is not None:
                validated_entry["range_result"] = previous_entry["range_result"]
            result_store[key] = validated_entry
        elif lockbox_clicked:
            previous_entry = result_store.get(key, {})
            lockbox_entry = run_lockbox_action(symbol, timeframe)
            if previous_entry.get("result") is not None:
                lockbox_entry["result"] = previous_entry["result"]
            if previous_entry.get("range_result") is not None:
                lockbox_entry["range_result"] = previous_entry["range_result"]
            result_store[key] = lockbox_entry
    except Exception as exc:
        operation_log = getattr(
            exc,
            "operation_log",
            result_store.get(key, {}).get("operation_log", ""),
        )
        show_error(exc, operation_log)
        return

    entry = result_store.get(key)
    if entry is None:
        st.info(
            "왼쪽에서 코인과 시간봉을 선택한 뒤 ‘최신 예측 확인’을 누르세요. "
            "전체 Walk-Forward 검증은 별도 버튼으로만 실행됩니다."
        )
        return

    raw_df = entry.get("raw_df")
    if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
        st.warning("화면에 표시할 종료 캔들 데이터가 없습니다.")
        return

    prediction_result = entry.get("result")
    render_recent_data(raw_df, prediction_result)
    if prediction_result is not None:
        render_prediction(prediction_result)
        render_candle_insight(prediction_result, entry.get("range_result"))
    else:
        st.info("최신 확률과 신호를 보려면 ‘최신 예측 확인’을 누르세요.")
    render_validation_summary(entry.get("validation_summary"), symbol, timeframe)
    render_model_info(symbol, timeframe, raw_df, prediction_result)

    with st.expander("최근 실행 기록"):
        operation_log = entry.get("operation_log", "").strip()
        st.code(operation_log or "별도 실행 기록이 없습니다.")

    if entry.get("notice"):
        st.markdown(
            f'<div class="footer-notice">{html.escape(str(entry["notice"]))}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        "연구 및 모의검증 용도입니다. 실제 주문 기능은 없으며, 비용 반영 전략 "
        "성과는 3배 레버리지 기준입니다. 표시된 확률과 과거 성과는 미래 결과를 "
        "보장하지 않습니다."
    )


if __name__ == "__main__":
    main()
