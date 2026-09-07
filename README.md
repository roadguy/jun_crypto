# HTML 암호화폐 예측 웹사이트 — 수정본

HTML·CSS·JavaScript 화면과 FastAPI Python 서버를 한 폴더로 제공합니다.
기존 방향 Model A, Q10/Q50/Q90 가격 구간, Walk-Forward, 최종 검증 기능을 포함합니다.
원본 파일·기존 CSV·저장 모델은 수정하지 않았습니다.

## Mac에서 실행

1. 압축을 풀고 `crypto_html_site_fixed` 폴더를 엽니다.
2. `start.command`를 실행합니다. 최초 실행은 패키지 설치 시간이 필요합니다.
3. 터미널에 서버 실행 메시지가 뜨면 **http://127.0.0.1:8000** 으로 접속합니다.
4. `최신 예측 확인`을 누릅니다. 최초 실행은 과거 캔들을 수집하고 모델을 학습합니다.
5. 종료하려면 실행 중인 터미널에서 Control+C를 누릅니다.

macOS에서 더블클릭 실행이 안 되면 해당 폴더의 터미널에서 `bash start.command`를 실행하세요.
실행기는 Python 3.12 또는 3.11을 찾습니다. Codex에 설치된 호환 Python이 있으면 사용할 수 있습니다.
Python 3.14에는 이 고정 버전 패키지 조합을 설치하지 마세요. Python 3.12에서 검증했습니다.
`libomp` 오류가 나오면 macOS Homebrew에서 `brew install libomp` 후 다시 실행합니다.

**`static/index.html`을 직접 더블클릭하거나 VS Code Live Server로 열면 Python API가 작동하지 않습니다.**
HTML 파일만 정적 호스팅에 올리는 것도 충분하지 않습니다.

수동 실행:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```

## 무엇이 바뀌었나

- Binance USD-M 공개 REST에서 시장 목록과 캔들만 직접 요청합니다. 거래소 전체 메타데이터 파싱에 의존하지 않습니다.
- 연결 5초·읽기 15초 제한, 일시 장애 최대 3회 재시도, 지역/IP 제한과 요청 제한의 구체적인 오류를 제공합니다.
- 403/451, 418/429는 반복 호출하지 않습니다. 429 응답의 Retry-After를 안내합니다.
- 시장 목록은 5분 캐시합니다. 이미 최신인 CSV는 시장 목록 요청 없이 재사용합니다.
- 전체 수집 중 체크포인트를 보존하고, 이후에는 새로 종료된 캔들만 추가합니다.
- 미완성 봉 제외, 결측·무한대·잘못된 고저가 검증을 적용합니다. 네트워크 장애 때 기존 CSV를 덮어쓰지 않습니다.
- 작업별 진행 로그를 실시간 표시합니다. 전역 stdout을 가로채 다른 요청의 출력을 섞지 않습니다.
- 가격 구간 계산 실패가 성공한 방향 예측을 지우지 않습니다. 실패한 구간은 빈 값과 경고로 표시합니다.
- 잘못된 코인·시간봉 입력을 422 응답으로 처리합니다.
- 새로고침 후 같은 탭의 진행 중 작업에 재연결합니다. 서버가 재시작되어 작업이 없어지면 재실행을 안내합니다.
- 계산 중 코인 변경을 막고, 선택 변경 시 이전 결과를 초기화합니다.
- 누락 지표를 0%로 표시하던 문제와 차트 크기 감시자 누적을 수정했습니다.
- 차트는 기본 최근 2,000봉만 전송합니다. `전체 저장 캔들 보기`로 전체 이력을 볼 수 있습니다. 지표는 전체 이력으로 계산합니다.
- 차트 라이브러리를 로컬에 포함해 외부 CDN 실패로 차트가 사라지는 문제를 줄였습니다.

## 데이터와 예측 시각

여기서 최신 데이터는 **종료된 캔들**입니다. 초 단위 체결가를 스트리밍하는 서비스가 아닙니다.
예측/새로고침 버튼을 누르면 새 종료봉을 확인합니다. 미완성 봉은 학습에 넣지 않습니다.
기준 캔들과 예측 대상 시각은 한국 시간으로 화면에 표시합니다.
오래된 캔들이 남아 있으면 예측 결과에 경고를 표시합니다.
저장된 모델은 기존 재학습 주기에 따라 재사용합니다. 모델 성능이 개선되었다고 주장하지 않습니다.

`data/`는 실행 시 자동 생성됩니다. 다른 위치는 `CRYPTO_XGB_DATA_ROOT`로 지정할 수 있습니다.
기존 CSV를 재사용하려면 기존 `data/`를 복사하거나 해당 환경변수를 설정하세요.
다른 패키지 버전에서 만든 모델은 호환성 문제가 날 수 있으므로 새 수정본에서는 처음부터 학습하는 것을 권장합니다.
`portfolio_results/`는 기존 과거 검증 결과의 표시용 복사본입니다. 현재 실시간 검증 결과가 아닙니다.

## 배포

Render에 올릴 때는 `RENDER_DEPLOY.md`의 순서를 따라 주세요.

`render.yaml`을 포함했습니다. Python 서버를 실행할 수 있는 호스팅이 필요합니다.
배포 명령: `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1`
Health check: `/api/health`. 작업 상태가 프로세스 메모리에 있으므로 worker는 **1개**로 유지합니다.
프로세스 재시작 시 진행 중 작업 정보는 사라지지만 저장된 체크포인트로 수집을 이어갈 수 있습니다.
모델과 CSV 유지가 필요하면 영구 저장소에 `CRYPTO_XGB_DATA_ROOT`를 연결하세요.
배포 서버에서 Binance가 403/451을 반환하면 코드만으로 해결되지 않습니다. 그 서버의 실제 API 접근 여부를 확인해야 합니다.
제공된 Singapore 설정도 접근 성공을 보장하지 않습니다.

이 수정본은 로컬에서 검증한 패키지이며 공개 서버에 배포한 상태는 아닙니다.
Sites의 Cloudflare Workers 실행 환경은 이 Python/XGBoost 서버 구조와 호환되지 않아 그곳에는 배포하지 않았습니다.

## 테스트

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

테스트는 별도 임시 데이터 폴더를 사용하고 실제 주문을 보내지 않습니다.
실제 API 검증 내역은 `VERIFICATION.md`를 참고하세요.

## 공개 API 참고

- [Binance 공식 Python 선물 커넥터의 시장 데이터 구현](https://github.com/binance/binance-futures-connector-python/blob/main/binance/um_futures/market.py)
- 사용 경로: `/fapi/v1/exchangeInfo`, `/fapi/v1/klines`
- 차트 라이브러리: Lightweight Charts 4.2.3. 라이선스는 `static/LICENSE.lightweight-charts.txt`에 포함했습니다.

실제 주문 기능은 없으며, 표시된 확률과 과거 모의성과는 연구용입니다.
