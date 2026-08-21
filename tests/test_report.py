"""Tests for the Gemini report request boundary.

These tests mock `google.genai.Client` entirely, so no real network call is ever
made -- they exercise how `generate_ai_report()` configures the request and reacts
to responses/errors, not the live Gemini API.
"""

import os
import unittest
from unittest.mock import patch

from google.genai import errors

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

        retry_options = http_options.retry_options
        self.assertEqual(retry_options.attempts, report.GEMINI_MAX_ATTEMPTS)
        self.assertEqual(retry_options.attempts, 3)  # bounded: no unlimited/long retry
        self.assertEqual(retry_options.initial_delay, report.GEMINI_RETRY_INITIAL_DELAY)
        self.assertEqual(retry_options.max_delay, report.GEMINI_RETRY_MAX_DELAY)
        self.assertEqual(retry_options.exp_base, report.GEMINI_RETRY_EXP_BASE)
        self.assertEqual(retry_options.http_status_codes, report.GEMINI_RETRY_STATUS_CODES)
        self.assertEqual(retry_options.http_status_codes, [429, 503])  # transient errors only

    @patch("google.genai.Client")
    def test_raises_on_empty_response(self, client_cls) -> None:
        client_cls.return_value.models.generate_content.return_value.text = None

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with self.assertRaises(report.EmptyReportError):
                report.generate_ai_report({"target": {"line": "LINE-B"}})

    @patch("google.genai.Client")
    def test_raises_friendly_error_on_persistent_503(self, client_cls) -> None:
        server_error = errors.ServerError(
            503,
            {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}},
        )
        client_cls.return_value.models.generate_content.side_effect = server_error

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with self.assertRaises(report.GeminiUnavailableError) as ctx:
                report.generate_ai_report({"target": {"line": "LINE-B"}})

        message = str(ctx.exception)
        self.assertIn("잠시 후 다시 시도", message)
        # the raw provider error JSON must not leak into the user-facing message
        self.assertNotIn("error", message)
        self.assertNotIn("UNAVAILABLE", message)


if __name__ == "__main__":
    unittest.main()
