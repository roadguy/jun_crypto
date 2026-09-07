# Render 배포 안내

## 가장 간단한 방법: Blueprint

1. 압축을 풀고 `crypto_html_site_fixed` **폴더 안의 파일들**을 GitHub 저장소 최상단에 올립니다. `app.py`, `requirements.txt`, `render.yaml`, `static/`이 같은 위치에 있어야 합니다.
2. Render에서 **New → Blueprint**를 선택하고 해당 저장소를 연결합니다.
3. `render.yaml`에 정의된 Python Web Service를 확인한 뒤 생성합니다. 기본 파일은 Free 요금제로 설정했습니다.
4. 배포가 완료되면 Render의 `https://…onrender.com` 주소를 엽니다.
5. `데이터 새로고침`으로 실제 거래소 연결부터 확인한 뒤 `최신 예측 확인`을 실행합니다.

`.venv`, `__pycache__`, 기존 학습 모델은 올리지 않습니다. `static/`만 따로 배포하지 마세요.

## 기존 서비스를 직접 설정할 때

| 항목 | 값 |
|---|---|
| Service type | Web Service |
| Runtime | Python 3 |
| Region | Singapore |
| Root Directory | 파일을 저장소 최상단에 올렸으면 비움 |
| Build Command | `python -m pip install -r requirements.txt` |
| Start Command | `python -m uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health Check Path | `/api/health` |

저장소 안에 `crypto_html_site_fixed` 폴더째 올렸다면 Root Directory를 `crypto_html_site_fixed`로 지정합니다.
Blueprint 방식은 저장소 최상단에 파일을 두는 구성을 기준으로 작성했습니다.

환경변수:

```text
PYTHON_VERSION=3.12.8
PYTHONUNBUFFERED=1
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
CRYPTO_XGB_DATA_ROOT=./data
```

`PORT`는 Render가 제공합니다. API 키는 필요하지 않습니다.
기존 서비스에 다른 PYTHON_VERSION이 설정되어 있으면 변경하세요.
로컬 실행용 `start.command`는 Render 시작 명령으로 사용하지 않습니다.

## 저장 데이터 유지

Free 서비스는 영구 디스크를 사용할 수 없습니다. 재배포·재시작 시 모델과 CSV가 사라져 다시 수집·학습할 수 있습니다.
상시 운영하며 데이터를 유지하려면 디스크를 지원하는 유료 서비스에서 디스크를 추가하고, 예를 들어 `/var/data`에 마운트한 뒤 다음 환경변수를 지정합니다.

```text
CRYPTO_XGB_DATA_ROOT=/var/data/crypto_xgb
```

현재 render.yaml에는 유료 서비스나 유료 디스크를 자동 생성하는 설정을 넣지 않았습니다.
무료 인스턴스에서 장시간 검증·학습의 메모리 한도 내 동작을 보장하지 않습니다. 메모리 부족으로 프로세스가 종료되면 인스턴스 메모리를 늘리거나 검증은 로컬에서 실행하세요.

## 배포 후 확인

- `/api/health`의 `status: ok`는 앱 실행 확인입니다. Binance 연결 성공까지 뜻하지는 않습니다.
- `데이터 새로고침`의 완료 메시지와 차트를 확인합니다.
- 403/451이면 해당 Render 서버 IP·지역의 Binance 접근 제한입니다. Singapore도 성공을 보장하지 않습니다.
- 429/418이면 화면에 표시된 대기 시간을 확인합니다.
- `ModuleNotFoundError`이면 Root Directory·Build Command·설치 로그를 확인합니다.
- `No open ports detected`이면 위 Start Command와 `$PORT` 설정을 확인합니다.
- 모델 계산 중 서버가 재시작되면 브라우저에서 재실행해야 합니다. 보존된 다운로드 체크포인트가 있으면 이어받습니다.

이 패키지를 Render에 실제 배포한 상태는 아닙니다. 배포 성공 여부와 해당 서버의 실시간 API 접근은 배포 후 확인해야 합니다.

공식 문서: [FastAPI 배포](https://render.com/docs/deploy-fastapi), [Python 버전](https://render.com/docs/python-version), [무료 서비스 제한](https://render.com/docs/free).
