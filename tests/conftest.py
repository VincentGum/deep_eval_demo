"""Pytest configuration for customer agent tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure src is in path
_SRC_PATH = Path(__file__).parent.parent / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Disable DeepEval telemetry
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    # Set Python path for imports
    os.environ.setdefault("PYTHONPATH", str(_SRC_PATH))

    yield

    # Cleanup (if needed)
    pass


@pytest.fixture
def mock_human_approval():
    """Provide a mock human approval function for testing."""

    def approve_or_reject(user_message: str, draft: str) -> tuple[bool, str | None]:
        """Mock human approval function.

        Args:
            user_message: Original user message
            draft: Proposed response from agent

        Returns:
            Tuple of (approved, modified_response)
        """
        # For demo: auto-approve all
        return True, None

    return approve_or_reject


@pytest.fixture
def strict_mock_human_approval():
    """Provide a strict mock human approval function for testing rejection."""

    def reject_with_modification(user_message: str, draft: str) -> tuple[bool, str]:
        """Mock human approval that always rejects and modifies.

        Args:
            user_message: Original user message
            draft: Proposed response from agent

        Returns:
            Tuple of (approved, modified_response)
        """
        modified = draft + "\n\n[Human Modified: Your request has been escalated.]"
        return False, modified

    return reject_with_modification
