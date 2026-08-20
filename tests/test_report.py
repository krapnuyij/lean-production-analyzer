"""Tests for the Gemini report request boundary."""

import os
import unittest
from unittest.mock import patch

from src import report


class GenerateAIReportTest(unittest.TestCase):
    @patch("google.genai.Client")
    def test_configures_bounded_request(self, client_cls) -> None:
        client_cls.return_value.models.generate_content.return_value.text = "generated report"

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            result = report.generate_ai_report({"target": {"line": "LINE-B"}})

        self.assertEqual(result, "generated report")
        http_options = client_cls.call_args.kwargs["http_options"]
        self.assertEqual(http_options.timeout, report.GEMINI_REQUEST_TIMEOUT_MS)
        self.assertEqual(http_options.retry_options.attempts, report.GEMINI_MAX_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
