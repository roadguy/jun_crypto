# Render 배포 설정

이 저장소의 `render.yaml`은 FastAPI 서버를 실행하는 설정입니다. HTML 파일만 올리는 정적 호스팅으로는 모델을 실행할 수 없습니다.

## 저장소 연결

1. Render에서 저장소를 연결해 Blueprint를 생성합니다.
2. `render.yaml`의 서비스 설정을 확인합니다.
3. 배포 후 서비스 주소를 열고 **데이터 새로고침**으로 Binance 연결을 확인합니다.
4. 데이터 수집에 성공하면 **최신 예측 확인**을 실행합니다.

수동으로 Web Service를 만들 때는 다음 값을 사용합니다.

| 항목 | 값 |
| --- | --- |
| Root Directory | 저장소 최상단 |
| Build Command | `python -m pip install -r requirements.txt` |
| Start Command | `python -m uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health Check Path | `/api/health` |

`PORT`는 호스팅 환경에서 제공합니다. Python 버전과 수치 연산 스레드 설정은 `render.yaml`을 기준으로 맞춥니다. 로컬용 `start.command`는 배포 시작 명령으로 사용하지 않습니다.

## 데이터와 작업 상태

CSV와 모델은 `CRYPTO_XGB_DATA_ROOT`에 저장합니다. 현재 설정은 `./data`입니다. 재시작 후에도 보존하려면 영구 저장소를 연결하고 그 경로를 환경변수에 지정해야 합니다.

작업 상태는 프로세스 메모리에 있어 **worker를 1개로 실행**합니다. 서버가 재시작되면 진행 중 작업 정보는 사라집니다. 저장된 다운로드 체크포인트가 있으면 수집을 이어갈 수 있습니다.

장시간 학습과 검증은 메모리를 많이 사용할 수 있습니다. 배포 설정 파일이 있다는 것만으로 해당 서버에서 학습이나 거래소 접근에 성공했다는 뜻은 아닙니다.

## 배포 후 확인

- `/api/health`의 `status: ok`: 앱이 실행 중이라는 뜻입니다. 거래소 접속까지 확인하지는 않습니다.
- Binance HTTP 403·451: 서버 IP 또는 지역의 접근 제한을 확인합니다. 설정된 Singapore 지역도 접근 성공을 보장하지 않습니다.
- HTTP 418·429: 화면에 안내된 대기 시간을 확인합니다.
- `ModuleNotFoundError`: 설치 로그와 Root Directory를 확인합니다.
- 포트 감지 실패: Start Command의 `--host 0.0.0.0`과 `$PORT`를 확인합니다.
- 계산 중 서버 종료: 메모리 사용량과 서버 로그를 확인합니다.

호스팅 설정의 세부 사항은 [Render의 FastAPI 배포 문서](https://render.com/docs/deploy-fastapi)를 참고하세요.
