"""회귀 방지용 핵심 검증 테스트.

네트워크나 실제 Binance 데이터 없이 타깃 정렬, 누수 방지, Lockbox 분리,
포지션 유지/전환 수수료와 누적 자산 계산을 확인한다.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import crypto_xgb_core as core
import crypto_xgb_validation as validation
import crypto_xgb_range_research as range_research


def prediction_frame(closes, returns, probabilities, threshold=0.56):
    count = len(closes)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=count, freq="4h", tz="UTC"),
            "close": closes,
            "future_return": returns,
            "target_up": (np.asarray(returns) > 0).astype(int),
            "probability_up": probabilities,
            "threshold": threshold,
        }
    )


class BacktestAccountingTests(unittest.TestCase):
    def test_bankruptcy_stops_a_position_before_price_recovery(self):
        frame = prediction_frame(
            closes=[100.0, 60.0, 120.0],
            returns=[-0.40, 1.0, 0.10],
            probabilities=[0.70, 0.70, 0.70],
        )
        result = validation._simulate_fixed_position_strategy(frame, "4h")
        np.testing.assert_array_equal(result["equity"], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result["position"], [1, 0, 0])
        self.assertTrue(result["bankrupt"])
        self.assertEqual(result["trade_returns"].tolist(), [-1.0])

    def test_empty_backtest_has_a_clear_error(self):
        frame = prediction_frame([], [], [])
        with self.assertRaisesRegex(ValueError, "예측 기록"):
            validation._simulate_fixed_position_strategy(frame, "4h")

    def test_same_position_is_not_charged_each_bar(self):
        frame = prediction_frame(
            closes=[100.0, 110.0], returns=[0.10, 0.10], probabilities=[0.70, 0.70]
        )
        result = validation._simulate_fixed_position_strategy(frame, "4h")

        # 3배 고정 계약: 100에서 0.03 BTC, 121 청산.
        # 진입 수수료 3*0.02%=0.0006, 청산 수수료 3.63*0.02%=0.000726.
        self.assertEqual(len(result["trade_returns"]), 1)
        self.assertAlmostEqual(result["total_fee"], 0.001326, places=9)
        self.assertAlmostEqual(result["equity"][-1], 1.628674, places=9)

    def test_neutral_to_long_has_one_entry_and_one_exit(self):
        frame = prediction_frame(
            closes=[100.0, 100.0], returns=[0.0, 0.01], probabilities=[0.50, 0.70]
        )
        result = validation._simulate_fixed_position_strategy(frame, "4h")
        expected_fee = 3.0 * 0.0002 + 3.03 * 0.0002
        self.assertEqual(len(result["trade_returns"]), 1)
        self.assertAlmostEqual(result["total_fee"], expected_fee, places=9)

    def test_long_to_short_charges_close_and_reentry(self):
        frame = prediction_frame(
            closes=[100.0, 101.0], returns=[0.01, -0.01], probabilities=[0.70, 0.30]
        )
        result = validation._simulate_fixed_position_strategy(frame, "4h")
        self.assertEqual(len(result["trade_returns"]), 2)
        # 두 거래가 각각 진입·청산되므로 편도 비용이 총 네 번 발생한다.
        self.assertGreater(result["total_fee"], 4 * 0.0005)


class FeatureAndSplitTests(unittest.TestCase):
    @staticmethod
    def raw_frame(rows=800):
        rng = np.random.default_rng(20260817)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, rows)))
        open_price = np.r_[close[0], close[:-1]]
        spread = rng.uniform(0.001, 0.006, rows)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC"),
                "open": open_price,
                "high": np.maximum(open_price, close) * (1 + spread),
                "low": np.minimum(open_price, close) * (1 - spread),
                "close": close,
                "volume": rng.lognormal(8, 0.4, rows),
            }
        )

    def test_target_uses_exactly_next_close(self):
        raw = self.raw_frame()
        featured = core.add_features(raw)
        index = 300
        expected = raw.loc[index + 1, "close"] / raw.loc[index, "close"] - 1
        self.assertAlmostEqual(featured.loc[index, "future_return"], expected, places=14)

    def test_future_price_change_does_not_change_past_features(self):
        raw = self.raw_frame()
        changed = raw.copy()
        cutoff = 350
        changed.loc[cutoff + 1 :, ["open", "high", "low", "close", "volume"]] *= 7
        original_features = core.add_features(raw).loc[:cutoff, core.FEATURE_COLUMNS]
        changed_features = core.add_features(changed).loc[:cutoff, core.FEATURE_COLUMNS]
        pd.testing.assert_frame_equal(original_features, changed_features)

    def test_latest_ninety_days_are_reserved_and_purged(self):
        raw = self.raw_frame()
        model = core.make_model_data(core.add_features(raw))
        development, lockbox, metadata = validation.split_development_lockbox(model, 90)
        self.assertGreaterEqual(len(lockbox), 89)
        self.assertLess(development["timestamp"].max(), lockbox["timestamp"].min())
        self.assertEqual(metadata["purge_bars"], core.PURGE_BARS)

    def test_lockbox_cannot_overlap_development_with_short_history(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=500, freq="h", tz="UTC")
        })
        with self.assertRaisesRegex(ValueError, "데이터가 부족"):
            validation.split_development_lockbox(frame, 90)

    def test_range_holdout_cannot_overlap_development_with_short_history(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=500, freq="h", tz="UTC")
        })
        design = {
            "holdout_start": "2025-12-01T00:00:00+00:00",
            "holdout_end": "2026-02-01T00:00:00+00:00",
            "timeframe": "1h",
        }
        with self.assertRaisesRegex(ValueError, "개발 데이터가 부족"):
            range_research.split_development_holdout(frame, design)

    def test_invalid_latest_features_do_not_fall_back_to_older_candle(self):
        featured = core.add_features(self.raw_frame())
        featured.loc[featured.index[-1], "rsi_14"] = np.nan
        with self.assertRaisesRegex(ValueError, "최신 종료 캔들"):
            core.latest_feature_row(featured)

    def test_one_class_calibration_uses_raw_probability(self):
        calibrators = core.fit_calibrators([1, 1, 1], [0.4, 0.6, 0.8])
        self.assertEqual(calibrators, {"raw": None})


class DiagnosticTests(unittest.TestCase):
    def test_block_bootstrap_auc_returns_ordered_interval(self):
        rng = np.random.default_rng(7)
        y = rng.integers(0, 2, 500)
        probability = np.clip(0.45 + 0.1 * y + rng.normal(0, 0.1, 500), 0.01, 0.99)
        result = validation.block_bootstrap_auc(y, probability, repeats=50)
        self.assertEqual(result["status"], "AVAILABLE")
        self.assertLessEqual(result["auc_ci_95_low"], result["auc_ci_95_high"])


if __name__ == "__main__":
    unittest.main()
