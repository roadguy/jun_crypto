#!/bin/bash
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [ ! -x .venv/bin/python ]; then
  selected_python=""
  for candidate in "${PYTHON_BIN:-python3.12}" python3.11 python3 "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"; do
    if "$candidate" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13)' 2>/dev/null; then
      selected_python="$candidate"
      break
    fi
  done
  if [ -z "$selected_python" ]; then
    echo 'Python 3.12를 설치한 뒤 다시 실행하세요. 이 패키지는 Python 3.11~3.12용입니다.'
    read -r -p 'Enter를 누르면 종료합니다.'
    exit 1
  fi
  "$selected_python" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13), "기존 .venv의 Python 버전이 맞지 않습니다. 다른 이름으로 보관하고 다시 실행하세요."'
.venv/bin/python -m pip install -r requirements.txt
printf '\n브라우저에서 http://127.0.0.1:%s 를 여세요. 종료: Control+C\n' "${PORT:-8000}"
exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port "${PORT:-8000}" --workers 1
