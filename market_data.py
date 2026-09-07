"""Binance USD-M public REST only: no keys, orders, spot, or options discovery."""
from __future__ import annotations

import time
import threading
import requests
import ccxt


class BinancePublicData:
    rateLimit = 250
    _markets = None
    _markets_at = 0.0
    _lock = threading.Lock()

    def __init__(self):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'SignalLedger/1.1'

    @staticmethod
    def milliseconds():
        return int(time.time() * 1000)

    @staticmethod
    def parse_timeframe(timeframe):
        return {'15m': 900, '1h': 3600, '4h': 14400, '1d': 86400}[timeframe]

    def _get(self, endpoint, params=None):
        try:
            response = self.session.get('https://fapi.binance.com' + endpoint,
                                        params=params, timeout=(5, 15))
        except requests.exceptions.SSLError as exc:
            raise ValueError('Binance TLS 인증서 검증 실패. 인증서와 네트워크 설정을 확인하세요.') from exc
        except requests.exceptions.Timeout as exc:
            raise ccxt.RequestTimeout('Binance 응답 시간 초과 (연결 5초 / 읽기 15초)') from exc
        except requests.exceptions.ConnectionError as exc:
            raise ccxt.NetworkError('Binance 연결 실패. DNS·방화벽·인터넷 연결을 확인하세요.') from exc
        status = response.status_code
        if status in (403, 451):
            raise ValueError(f'Binance HTTP {status}: 서버 IP 또는 지역에서 접근이 제한되었습니다. 배포 서버의 접근 가능 여부를 확인하세요.')
        if status in (418, 429):
            # Do not retry an IP ban immediately or ignore Retry-After.
            raise ValueError(f'Binance HTTP {status}: 요청 제한. {response.headers.get("Retry-After", "60")}초 후 다시 시도하세요.')
        if status >= 500:
            raise ccxt.ExchangeNotAvailable(f'Binance HTTP {status}: 일시적인 서버 오류')
        if status != 200:
            raise ValueError(f'Binance HTTP {status}: 공개 데이터 요청 거절 ({endpoint})')
        try:
            payload = response.json()
        except ValueError as exc:
            raise ccxt.ExchangeNotAvailable('Binance에서 JSON 대신 잘못된 응답을 받았습니다.') from exc
        if isinstance(payload, dict) and payload.get('code', 0) < 0:
            raise ValueError(f'Binance {payload["code"]}: {payload.get("msg", "요청 실패")}')
        return payload

    def load_markets(self):
        cls = type(self)
        with cls._lock:
            if cls._markets is not None and time.monotonic() - cls._markets_at < 300:
                return cls._markets
            payload = self._get('/fapi/v1/exchangeInfo')
            if not isinstance(payload, dict) or not isinstance(payload.get('symbols'), list):
                raise ccxt.ExchangeNotAvailable('Binance 시장 목록 형식이 올바르지 않습니다.')
            markets = {}
            for item in payload['symbols']:
                if item.get('contractType') != 'PERPETUAL' or item.get('quoteAsset') != 'USDT' or item.get('marginAsset') != 'USDT':
                    continue
                symbol = f'{item["baseAsset"]}/USDT:USDT'
                markets[symbol] = {'swap': True, 'linear': True, 'quote': 'USDT',
                                   'settle': 'USDT', 'active': item.get('status') == 'TRADING'}
            cls._markets, cls._markets_at = markets, time.monotonic()
            return markets

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        payload = self._get('/fapi/v1/klines', {
            'symbol': symbol.split('/')[0] + 'USDT', 'interval': timeframe,
            'startTime': int(since), 'limit': int(limit),
        })
        if not isinstance(payload, list):
            raise ccxt.ExchangeNotAvailable('Binance 캔들 응답 형식이 올바르지 않습니다.')
        try:
            rows = [[int(row[0]), *[float(v) for v in row[1:6]]] for row in payload]
            if any(len(row) != 6 for row in rows):
                raise ValueError('missing OHLCV')
            return sorted((row for row in rows if row[0] >= since), key=lambda row: row[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise ccxt.ExchangeNotAvailable('Binance 캔들 값이 올바르지 않습니다.') from exc
