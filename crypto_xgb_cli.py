"""코인과 시간봉을 선택하면 XGBoost Model A 예측을 출력하는 프로그램."""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stderr, redirect_stdout

import pandas as pd

import crypto_xgb_core as core
import crypto_xgb_validation as validation


TIMEFRAME_OPTIONS = [
    ("15분", "15m"),
    ("1시간", "1h"),
    ("4시간", "4h"),
    ("1일", "1d"),
]
TIMEFRAME_LABELS = {value: label for label, value in TIMEFRAME_OPTIONS}


def input_coin(input_func=input):
    """임의의 토큰명을 입력받아 USD-M USDT 선물 심볼로 변환한다."""
    while True:
        print("\n[코인 입력]")
        try:
            token = input_func("토큰명 (예: BTC, ETH, XRP, 1000PEPE): ")
        except (EOFError, KeyboardInterrupt):
            print("\n입력이 중단되었습니다.")
            return None
        try:
            return core.symbol_from_token(token)
        except ValueError as exc:
            print(exc)


def choose_menu(title: str, options, input_func=input):
    """번호가 잘못되면 종료하지 않고 다시 입력받는다."""
    while True:
        print(f"\n[{title}]")
        for index, (label, _) in enumerate(options, start=1):
            print(f"{index}) {label}")
        try:
            value = input_func("선택: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n입력이 중단되었습니다.")
            return None
        if value.isdigit() and 1 <= int(value) <= len(options):
            return options[int(value) - 1][1]
        print(f"1부터 {len(options)} 사이의 번호를 입력하세요.")


def clear_screen() -> None:
    """선택 메뉴를 지워 최종 결과만 화면에 남긴다."""
    try:
        # VS Code의 Jupyter/Interactive Window에서 이전 셀 출력을 제거한다.
        from IPython import get_ipython
        from IPython.display import clear_output

        if get_ipython() is not None:
            clear_output(wait=True)
            return
    except ImportError:
        pass

    # 일반 터미널에서는 ANSI 제어문자로 화면을 지운다.
    print("\033[2J\033[H", end="", flush=True)


def print_recent_ohlcv(raw_df: pd.DataFrame) -> None:
    """최근 종료된 OHLCV 캔들 5개를 출력한다."""
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    recent = raw_df.sort_values("timestamp").tail(5)[columns].copy()

    print("\n[최근 종료된 OHLCV 캔들 5개]")
    print(
        recent.to_string(
            index=False,
            formatters={
                "timestamp": lambda value: str(value),
                "open": lambda value: f"{value:,.2f}",
                "high": lambda value: f"{value:,.2f}",
                "low": lambda value: f"{value:,.2f}",
                "close": lambda value: f"{value:,.2f}",
                "volume": lambda value: f"{value:,.6f}",
            },
        )
    )


def _metric(value, formatter) -> str:
    return "계산 불가" if value is None else formatter(value)


def reliability_status(summary: dict) -> str:
    """OOS 성능과 Fold 안정성을 함께 사용한 보수적 상태 표시다."""
    auc = float(summary.get("roc_auc", 0.5))
    fold_ratio = float(summary.get("folds_auc_above_0_5", 0.0))
    samples = int(
        summary.get(
            "oos_samples",
            summary.get("strategy", {}).get("prediction_opportunities", 0),
        )
    )
    if samples < 500 or int(summary.get("folds", 0)) < 3:
        return "표본 부족"
    if auc <= 0.50:
        return "무작위 수준 이하"
    if fold_ratio < 0.60:
        return "시장 구간별 성능 불안정"
    if auc < 0.53:
        return "매우 약한 예측력"
    if auc < 0.56:
        return "약한 예측력 / 관찰 필요"
    return "상대적으로 양호 / 지속 검증 필요"


def print_validation_summary(
    summary: dict | None,
    validation_error: str | None = None,
) -> None:
    """저장된 Walk-Forward 결과만 읽어 신뢰도와 전략 성과를 표시한다."""
    print("\n[모델 OOS 신뢰도]")
    if summary is None:
        print("검증 상태       : OOS 검증을 완료하지 못했습니다.")
        if validation_error:
            print(f"원인            : {validation_error}")
        else:
            print("원인            : 저장된 검증 결과가 없습니다.")
        return

    folds = int(summary.get("folds", 0))
    fold_ratio = float(summary.get("folds_auc_above_0_5", 0.0))
    folds_above = int(round(folds * fold_ratio))
    strategy = summary.get("strategy", {})
    oos_samples = int(
        summary.get("oos_samples", strategy.get("prediction_opportunities", 0))
    )
    generated = pd.to_datetime(summary.get("generated_at"), utc=True, errors="coerce")
    generated_text = (
        "알 수 없음"
        if pd.isna(generated)
        else generated.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M")
    )

    print(f"ROC-AUC         : {float(summary['roc_auc']):.4f}")
    print(f"Brier Score     : {float(summary['brier']):.4f}")
    print(f"Calibration gap : {float(summary['calibration_gap']):.4f}")
    print(f"Fold 평균 AUC   : {float(summary['fold_mean_auc']):.4f}")
    print(
        f"AUC>0.5 Fold    : {folds_above}/{folds} "
        f"({fold_ratio:.2%})"
    )
    print(f"OOS 표본        : {oos_samples:,}개")
    print(f"검증 시각       : {generated_text}")
    print(f"검증 상태       : {reliability_status(summary)}")

    print("\n[비용 반영 전략 성과]")
    print(f"종료 거래       : {int(strategy.get('closed_trades', 0)):,}회")
    print(
        "승률            : "
        + _metric(strategy.get("win_rate"), lambda value: f"{value:.2%}")
    )
    print(
        "손익비          : "
        + _metric(strategy.get("payoff_ratio"), lambda value: f"{value:.2f}")
    )
    print(
        "거래당 기대값   : "
        + _metric(strategy.get("expectancy"), lambda value: f"{value:.2%}")
    )
    print(
        f"누적수익률      : "
        f"{float(strategy.get('cumulative_return', 0.0)):.2%}"
    )
    print(
        "Sharpe          : "
        + _metric(strategy.get("sharpe"), lambda value: f"{value:.2f}")
    )
    print(f"MDD             : {float(strategy.get('mdd', 0.0)):.2%}")


def print_result(
    result: dict,
    raw_df: pd.DataFrame,
    validation_summary: dict | None,
    validation_error: str | None = None,
) -> None:
    """최근 OHLCV, 최신 예측, 저장된 OOS 검증 결과를 출력한다."""
    print_recent_ohlcv(raw_df)

    coin = result["symbol"].split("/")[0]
    timeframe_label = TIMEFRAME_LABELS[result["timeframe"]]
    print(f"\n[{coin} {timeframe_label} XGBoost Model A 최신 예측]")
    print(
        f"예측 대상      : {result['target_candle_start']} ~ "
        f"{result['target_candle_end']}"
    )
    print(f"기준 종가      : {result['reference_close']:,.2f}")
    print(f"상승 확률      : {result['probability_up']:.2%}")
    print(f"하락 확률      : {result['probability_down']:.2%}")
    print(f"확률 보정      : {result['calibration_method']}")
    print(f"거래 임계값    : {result['threshold']:.2%}")
    print(f"신호           : {result['signal']}")
    print_validation_summary(validation_summary, validation_error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="범용 암호화폐 XGBoost Model A 예측")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--symbol", help="토큰명 예: BTC, XRP, 1000PEPE")
    parser.add_argument("--timeframe", choices=list(core.TIMEFRAME_CONFIGS))
    # Jupyter가 자동으로 전달하는 --f=kernel.json 등의 인자는 무시한다.
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    if bool(args.symbol) != bool(args.timeframe):
        print("비대화형 실행은 --symbol과 --timeframe을 함께 지정해야 합니다.")
        return

    if args.symbol and args.timeframe:
        try:
            symbol = core.symbol_from_token(args.symbol)
        except ValueError as exc:
            print(exc)
            return
        timeframe = args.timeframe
    else:
        print("=" * 52)
        print("암호화폐 XGBoost 방향 예측 프로그램")
        print("=" * 52)
        symbol = input_coin()
        if symbol is None:
            return
        timeframe = choose_menu("시간봉 선택", TIMEFRAME_OPTIONS)
        if timeframe is None:
            return

    # 코인과 시간봉 선택 메뉴는 남기지 않고 결과 화면만 표시한다.
    clear_screen()

    validation_error = None
    try:
        # 코어에서 출력하는 수집·튜닝 진행 로그와 라이브러리 경고는 숨긴다.
        # 실행 결과 화면에는 OHLCV 5개와 최신 예측만 표시한다.
        hidden_output = io.StringIO()
        with redirect_stdout(hidden_output), redirect_stderr(hidden_output):
            result = core.run_latest_prediction(
                symbol=symbol,
                timeframe=timeframe,
                refresh=not args.no_refresh,
                force_retrain=False,
            )
            # 위에서 데이터 갱신이 끝났으므로 네트워크 요청 없이 읽는다.
            raw_df = core.load_raw_data(symbol, timeframe, refresh=False)
            validation_summary = validation.load_validation_summary(
                symbol, timeframe
            )
    except Exception as exc:
        print(f"\n[실행 실패] {exc}")
        return

    if validation_summary is None:
        coin = symbol.split("/")[0]
        timeframe_label = TIMEFRAME_LABELS[timeframe]
        print(
            f"[{coin} {timeframe_label} 최초 OOS Walk-Forward 검증 중...]\n"
            "처음 한 번은 시간이 오래 걸릴 수 있습니다."
        )
        try:
            hidden_output = io.StringIO()
            with redirect_stdout(hidden_output), redirect_stderr(hidden_output):
                validation_summary = validation.run_walk_forward(
                    symbol=symbol,
                    timeframe=timeframe,
                    refresh=False,
                )
        except Exception as exc:
            validation_error = str(exc)
        # 진행 문구를 지운 뒤 최종 결과만 남긴다.
        clear_screen()

    print_result(
        result,
        raw_df,
        validation_summary,
        validation_error,
    )


if __name__ == "__main__":
    main()
