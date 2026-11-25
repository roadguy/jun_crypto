"""작은 합성 데이터로 실제 모델 학습·저장·재사용 경로를 확인한다."""

import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

import crypto_xgb_core as core
import crypto_xgb_range_live as range_live
from tests import test_crypto_xgb_validation as fixtures


class PredictionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        environment = patch.dict(os.environ, CRYPTO_XGB_DATA_ROOT=temporary.name)
        environment.start()
        self.addCleanup(environment.stop)
        self.raw = fixtures.FeatureAndSplitTests.raw_frame(1200)
        self.raw["timestamp"] = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").floor("4h") - pd.Timedelta(hours=4),
            periods=len(self.raw), freq="4h",
        )
        self.symbol = "BTC/USDT:USDT"
        self.raw.to_csv(core.get_paths(self.symbol, "4h").raw_ohlcv, index=False)
        candidate = replace(core.XGB_CANDIDATES[0], n_estimators=12)
        configurations = patch.object(core, "XGB_CANDIDATES", [candidate])
        configurations.start()
        self.addCleanup(configurations.stop)

    def test_direction_training_save_and_reuse(self):
        first = core.run_latest_prediction(self.symbol, "4h", refresh=False)
        second = core.run_latest_prediction(self.symbol, "4h", refresh=False)
        self.assertTrue(0 < first["probability_up"] < 1)
        self.assertAlmostEqual(first["probability_up"] + first["probability_down"], 1)
        self.assertFalse(first["model_reused"])
        self.assertTrue(second["model_reused"])
        self.assertFalse(second["prediction_created"])
        log = core.load_paper_log(core.get_paths(self.symbol, "4h"))
        self.assertEqual(len(log), 1)
        self.assertEqual(pd.Timestamp(first["source_candle_start"]), self.raw["timestamp"].iloc[-1])

    def test_quantile_training_save_and_reuse(self):
        first = range_live.run_latest_range_prediction(self.symbol, "4h")
        second = range_live.run_latest_range_prediction(self.symbol, "4h")
        quantiles = [first["q10_price"], first["q50_price"], first["q90_price"]]
        self.assertTrue(np.isfinite(quantiles).all())
        self.assertEqual(quantiles, sorted(quantiles))
        self.assertFalse(first["model_reused"])
        self.assertTrue(second["model_reused"])

    def test_older_csv_does_not_return_a_later_logged_prediction(self):
        core.run_latest_prediction(self.symbol, "4h", refresh=False)
        older = self.raw.iloc[:-1]
        older.to_csv(core.get_paths(self.symbol, "4h").raw_ohlcv, index=False)
        result = core.run_latest_prediction(self.symbol, "4h", refresh=False)
        self.assertEqual(pd.Timestamp(result["source_candle_start"]), older["timestamp"].iloc[-1])
        self.assertFalse(result["model_reused"])

    def test_models_trained_on_later_data_are_not_reused(self):
        latest = self.raw["timestamp"].iloc[-1]
        artifact = {
            "model_version": core.MODEL_VERSION,
            "feature_columns": list(core.FEATURE_COLUMNS),
            "symbol": self.symbol,
            "timeframe": "4h",
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "latest_source_timestamp": (latest + pd.Timedelta(hours=4)).isoformat(),
        }
        self.assertFalse(core.artifact_is_current(artifact, latest, self.symbol, "4h"))
        artifact["model_version"] = range_live.RANGE_MODEL_VERSION
        self.assertFalse(range_live.artifact_is_current(artifact, self.symbol, "4h", latest))

    def test_future_training_time_is_not_reused(self):
        latest = self.raw["timestamp"].iloc[-1]
        artifact = {
            "model_version": core.MODEL_VERSION,
            "feature_columns": list(core.FEATURE_COLUMNS),
            "symbol": self.symbol,
            "timeframe": "4h",
            "trained_at": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).isoformat(),
            "latest_source_timestamp": latest.isoformat(),
        }
        self.assertFalse(core.artifact_is_current(artifact, latest, self.symbol, "4h"))
        artifact["model_version"] = range_live.RANGE_MODEL_VERSION
        self.assertFalse(range_live.artifact_is_current(artifact, self.symbol, "4h", latest))


if __name__ == "__main__":
    unittest.main()
