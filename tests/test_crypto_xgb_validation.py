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
