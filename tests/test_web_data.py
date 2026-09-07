import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import ccxt
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import app
import crypto_xgb_core as core
from market_data import BinancePublicData


class IsolatedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, CRYPTO_XGB_DATA_ROOT=self.tmp.name)
        self.env.start()
        self.client = TestClient(app.app)
    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()
    def frame(self, rows=300):
        end = pd.Timestamp.now(tz='UTC').floor('4h') - pd.Timedelta(hours=4)
        return pd.DataFrame({'timestamp': pd.date_range(end=end, periods=rows, freq='4h'),
                             'open': 100., 'high': 102., 'low': 98., 'close': 101., 'volume': 10.})


class APITests(IsolatedTest):
    def test_invalid_input_is_422(self):
        for url in ['/api/snapshot?token=../bad', '/api/chart?timeframe=5m']:
            self.assertEqual(self.client.get(url).status_code, 422)
    def test_lockbox_confirmation_checked_before_queue(self):
        response = self.client.post('/api/jobs', json={'action': 'lockbox'})
        self.assertEqual(response.status_code, 422)
    def test_empty_snapshot_does_not_fetch(self):
        with patch.object(core, 'fetch_ohlcv', side_effect=AssertionError('network')):
            response = self.client.get('/api/snapshot')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['has_data'])
    def test_chart_missing(self):
        self.assertEqual(self.client.get('/api/chart').status_code, 404)
    def test_chart_bounded_and_ema_keeps_history(self):
        raw = self.frame(3000)
        raw['close'] = np.linspace(100, 101, len(raw))
        payload = app._chart_payload(raw)
        self.assertEqual(payload['count'], 2000)
        self.assertEqual(payload['total_count'], 3000)
        self.assertEqual(len(payload['indicators']['ema200']), 2000)
    def test_serialization(self):
        value = app._clean({'paths': core.get_paths('BTC/USDT:USDT', '4h'), 'v': np.nan})
        import json
        json.dumps(value, allow_nan=False)
        self.assertIsNone(value['v'])
    def test_partial_prediction_survives_range_failure(self):
        job_id = 'partial-test'
        app.JOBS[job_id] = {'status': 'queued', 'log': ''}
        with patch.object(core, 'run_latest_prediction', return_value={'probability_up': .6}), \
             patch.object(app.range_live, 'run_latest_range_prediction', side_effect=ValueError('data 부족')), \
             patch.object(app, '_read_snapshot', return_value={'has_data': True, 'is_stale': False}):
            app._execute_job(job_id, app.JobRequest(action='predict'))
        job = app.JOBS.pop(job_id)
        self.assertEqual(job['status'], 'complete')
        self.assertIsNone(job['result']['range_prediction'])
        self.assertTrue(job['result']['warnings'])
        self.assertIn('방향 예측 완료', job['log'])
    def test_static_library_is_local(self):
        response = self.client.get('/static/lightweight-charts.js')
        self.assertEqual(response.status_code, 200)
    def test_duplicate_jobs_rejected(self):
        with patch.object(app.EXECUTOR, 'submit'):
            first = self.client.post('/api/jobs', json={'action': 'refresh'})
            second = self.client.post('/api/jobs', json={'action': 'refresh'})
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 409)
        app.JOBS.pop(first.json()['job_id'])


class DataTests(IsolatedTest):
    def test_latest_cache_needs_no_market_request(self):
        raw = self.frame()
        paths = core.get_paths('BTC/USDT:USDT', '4h')
        raw.to_csv(paths.raw_ohlcv, index=False)
        with patch.object(BinancePublicData, 'load_markets', side_effect=AssertionError('network')):
            result = core.load_raw_data('BTC/USDT:USDT', '4h')
        self.assertEqual(len(result), len(raw))
    def test_read_only_cache_does_not_wait_for_writer(self):
        raw = self.frame()
        raw.to_csv(core.get_paths('BTC/USDT:USDT', '4h').raw_ohlcv, index=False)
        with patch.object(core, 'FileLock', side_effect=AssertionError('reader blocked')):
            self.assertEqual(len(core.load_raw_data('BTC/USDT:USDT', '4h', refresh=False)), len(raw))
    def test_incomplete_candle_removed(self):
        raw = core._raw_candle_frame([[0,1,2,1,2,1],[14400000,1,2,1,2,1]])
        result = core._finalize_raw_frame(raw, 14400000, 20000000)
        self.assertEqual(len(result), 1)
    def test_nan_and_impossible_prices_rejected(self):
        for column, value in [('close', np.nan), ('volume', np.inf), ('high', 90)]:
            raw = self.frame()
            raw.loc[0, column] = value
            with self.assertRaises(ValueError): core.validate_ohlcv(raw, '4h')
    def test_timeout_retries_are_bounded(self):
        exchange = Mock()
        exchange.fetch_ohlcv.side_effect = ccxt.RequestTimeout('timeout')
        with patch.object(core.time, 'sleep'), self.assertRaises(RuntimeError):
            core._fetch_page_with_retry(exchange, 'BTC/USDT:USDT', '4h', 0)
        self.assertEqual(exchange.fetch_ohlcv.call_count, 3)
    def test_permanent_error_not_retried(self):
        exchange = Mock()
        exchange.fetch_ohlcv.side_effect = ValueError('HTTP 451')
        with self.assertRaises(ValueError): core._fetch_page_with_retry(exchange, 'BTC/USDT:USDT', '4h', 0)
        self.assertEqual(exchange.fetch_ohlcv.call_count, 1)
    def test_failed_download_preserves_cache(self):
        raw = self.frame()
        raw['timestamp'] -= pd.Timedelta(days=2)
        path = core.get_paths('BTC/USDT:USDT', '4h').raw_ohlcv
        raw.to_csv(path, index=False)
        before = path.read_bytes()
        with patch.object(core, '_validate_exchange_market'), patch.object(core, '_fetch_page_with_retry', side_effect=ValueError('HTTP 451')):
            with self.assertRaises(ValueError): core.load_raw_data('BTC/USDT:USDT', '4h')
        self.assertEqual(path.read_bytes(), before)
    def test_451_and_rate_limit_actionable(self):
        exchange = BinancePublicData()
        for status in [451, 403, 418, 429]:
            response = Mock(status_code=status, headers={'Retry-After': '120'})
            with patch.object(exchange.session, 'get', return_value=response):
                with self.assertRaisesRegex(ValueError, str(status)): exchange._get('/fapi/v1/klines')
    def test_checkpoint_resume_and_deduplicate(self):
        paths = core.get_paths('BTC/USDT:USDT', '4h')
        raw = core._raw_candle_frame([[0,1,2,1,2,1], [0,1,3,1,2,2]])
        core._save_download_checkpoint(raw, paths, 'BTC/USDT:USDT', '4h', 10, 0)
        result, start = core._load_download_checkpoint(paths, 'BTC/USDT:USDT', '4h', 10)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['high'], 3)
        self.assertEqual(start, 0)


if __name__ == '__main__': unittest.main()
