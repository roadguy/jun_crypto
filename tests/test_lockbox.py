"""이미 평가한 최종 구간은 버전 변경만으로 재평가하지 않는다."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import crypto_xgb_core as core
import crypto_xgb_validation as validation


class CachedLockboxTests(unittest.TestCase):
    def test_cached_evaluation_is_preserved_across_version_changes(self):
        for version in ["portfolio_diagnostics_v1", validation.DIAGNOSTIC_VERSION]:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as folder:
                with patch.dict(os.environ, CRYPTO_XGB_DATA_ROOT=folder):
                    path = core.get_paths("BTC/USDT:USDT", "4h").lockbox_summary
                    saved = {
                        "status": "EVALUATED_ONCE",
                        "model_version": core.MODEL_VERSION,
                        "diagnostic_version": version,
                        "feature_columns": list(core.FEATURE_COLUMNS),
                        "strategy": {"cumulative_return": 0.5},
                    }
                    path.write_text(json.dumps(saved), encoding="utf-8")
                    before = path.read_bytes()
                    with patch.object(core, "prepare_data", side_effect=AssertionError("재평가 금지")):
                        result = validation.evaluate_lockbox_once("BTC/USDT:USDT", "4h")
                    self.assertEqual(path.read_bytes(), before)
                    if version == validation.DIAGNOSTIC_VERSION:
                        self.assertEqual(result, saved)
                    else:
                        self.assertEqual(result["status"], "OUTDATED_DIAGNOSTICS")
                        self.assertNotIn("strategy", result)
                        self.assertIn("자동 재평가하지 않았습니다", result["warning"])


if __name__ == "__main__":
    unittest.main()
