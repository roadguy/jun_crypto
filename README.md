# jun_crypto

Binance의 가격·거래량 데이터를 수집하고, XGBoost로 다음 캔들의 상승 확률을 계산하는 개인 프로젝트입니다. 데이터 수집부터 모델 학습, 예측 결과를 웹 화면에 보여주는 과정까지 연결해봤습니다.

웹 화면 이름은 **Signal Ledger**입니다. 코인과 시간봉을 선택하면 가격 차트, 상승·하락 확률, 예상 가격 구간을 확인할 수 있습니다. 실제 주문을 실행하는 기능은 없습니다.

## 실행하기

Python **3.11 또는 3.12**가 필요합니다. 아래 명령은 저장소 폴더에서 실행합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```

브라우저에서 **http://127.0.0.1:8000**을 열고 다음 순서로 사용합니다.

1. 코인 이름과 시간봉을 선택합니다. 기본값은 BTC, 4시간봉입니다.
2. **데이터 새로고침**으로 캔들을 가져옵니다.
3. **최신 예측 확인**을 누르면 모델 학습과 예측을 진행합니다.

처음 실행할 때는 과거 데이터를 내려받고 방향·가격 구간 모델을 학습하므로 시간이 걸립니다. 이후에는 저장된 데이터와 모델을 재사용합니다. API 키는 필요하지 않지만 Binance 공개 API에 접속할 수 있어야 합니다.

macOS에서는 터미널에서 `bash start.command`로도 실행할 수 있습니다. 이 스크립트는 가상환경 생성과 의존성 설치를 함께 처리합니다. Windows에서는 가상환경 활성화 명령을 `.venv\Scripts\activate`로 바꿔주세요.

## 무엇을 예측하나요?

**마지막으로 종료된 캔들의 종가와 그다음 캔들의 종가를 비교**합니다. 예를 들어 4시간봉을 선택했다면 다음 4시간봉의 종가가 기준 종가보다 높을 확률을 계산합니다. 종가가 같으면 비상승으로 분류하므로, 화면의 ‘하락 확률’에는 보합도 포함됩니다.

| 항목 | 내용 |
| --- | --- |
| 데이터 | Binance USD-M USDT 무기한 선물의 OHLCV |
| 코인 | BTC·ETH·SOL·XRP 등 거래소에 상장된 USDT 무기한 선물 |
| 시간봉 | 15분, 1시간, 4시간, 1일 |
| 방향 예측 | XGBoost 분류 모델의 상승 확률과 보정된 확률 |
| 가격 구간 | 별도 분위수 회귀 모델의 Q10·Q50·Q90 |
| 화면 | 캔들 차트, 예측 시각, 확률, 가격 구간, 직접 실행한 검증 결과 |

미완성 캔들은 학습에서 제외합니다. 초 단위 실시간 시세가 아니라 버튼을 누를 때 갱신하는 종료 캔들을 사용합니다. 가격 구간은 다음 종가의 분포 추정값이며, 다음 캔들의 고가·저가를 뜻하지 않습니다.

## 처리 흐름

```text
Binance 공개 API → CSV 저장·검사 → 22개 피처 계산
                                ├─ 방향 분류 → 확률 보정 → 상승 확률
                                └─ 분위수 회귀 → Q10·Q50·Q90 가격
                                                    ↓
                                             FastAPI → 웹 화면
```

피처는 가격 변화, 캔들 모양, 거래량, 변동성, 이동평균과의 거리, 시간·요일로 구성했습니다. 학습과 검증은 시간을 섞지 않고 분리합니다. 피처 목록과 데이터 분할 방식은 [모델 설명](docs/MODEL.md)에 정리했습니다.

## 검증과 한계

웹 화면의 **고급 연구·검증**에서 Walk-Forward 백테스트를 별도로 실행할 수 있습니다. 과거 구간으로 학습한 뒤 그 이후 구간을 평가하는 방식이며, 일반 예측보다 오래 걸립니다. 마지막 90일은 개발 검증에서 제외하고 별도 평가용으로 남겨둡니다.

- 상승 확률이 높다고 수익이 보장되지는 않습니다. AUC 같은 예측 지표와 거래 성과는 따로 봐야 합니다.
- 백테스트는 3배 노출과 편도 수수료 0.02%를 가정합니다. 펀딩비, 슬리피지, 실제 체결 지연, 거래소의 청산 조건은 반영하지 않습니다.
- 가격 구간의 목표 포함률은 80%지만 실제 포함률이 항상 80%인 것은 아닙니다.
- 저장된 검증 결과는 특정 기간의 결과입니다. 현재 모델의 실시간 성과를 뜻하지 않습니다.

`portfolio_results/`에는 이전 실행의 요약 파일이 남아 있습니다. 원본 예측·데이터 전체가 포함되어 있지 않아 재현 결과로 제시하지 않으며, 웹 화면에도 자동으로 불러오지 않습니다.

## 파일 구성

| 파일 | 역할 |
| --- | --- |
| `market_data.py` | Binance 공개 REST 요청과 오류 처리 |
| `crypto_xgb_core.py` | 데이터 저장, 피처 생성, 방향 모델, 예측 기록 |
| `crypto_xgb_validation.py` | Walk-Forward·최종 평가와 백테스트 |
| `crypto_xgb_range_research.py` | 가격 구간 모델의 학습·평가 |
| `crypto_xgb_range_live.py` | 가격 구간 모델 저장과 최신 예측 |
| `app.py`, `static/` | FastAPI 서버와 웹 화면 |
| `crypto_xgb_cli.py` | 터미널에서 실행하는 방향 예측 |
| `crypto_xgb_web.py` | 이전 Streamlit 화면, 선택 설치 필요 |

실행 중 생성되는 CSV, 모델, 예측 기록은 `data/`에 저장됩니다. 다른 폴더를 쓰려면 `CRYPTO_XGB_DATA_ROOT` 환경변수를 지정합니다. 이 데이터와 `.venv/`는 Git 추적에서 제외합니다.

## 터미널 실행과 테스트

```bash
# 방향 예측. 검증은 자동으로 실행하지 않습니다.
python crypto_xgb_cli.py --symbol BTC --timeframe 4h

# 저장된 데이터로 예측하고 개발구간 검증도 실행
python crypto_xgb_cli.py --symbol BTC --timeframe 4h --no-refresh --validate

# 네트워크 없이 실행하는 자동 테스트
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
node tests/frontend.test.cjs
```

테스트 범위는 [VERIFICATION.md](VERIFICATION.md), 서버 배포 설정은 [RENDER_DEPLOY.md](RENDER_DEPLOY.md)를 참고하세요. 이전 Streamlit 화면은 `requirements_web.txt` 설치 후 `python -m streamlit run crypto_xgb_web.py`로 실행할 수 있습니다.

macOS에서 XGBoost를 불러올 때 `libomp` 오류가 나면 `brew install libomp`를 먼저 실행합니다. Binance HTTP 403·451은 접속 지역이나 서버 IP의 접근 제한, 418·429는 요청 제한입니다. 화면의 오류와 대기 시간을 확인해주세요.

기획과 구현 과정에서 AI 개발 도구를 활용했습니다. 차트는 [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts)를 사용하며, 배포 파일과 라이선스는 `static/`에 포함했습니다.
