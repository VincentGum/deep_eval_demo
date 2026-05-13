"""Human-in-the-Loop - Handles human input during task execution.

This module provides mechanisms for pausing task execution,
requesting human input, and resuming after input is received.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from .base import Task, TaskStatus


class HumanInputStatus(str, Enum):
    """Status of human input request."""
    PENDING = "pending"
    RECEIVED = "received"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class HumanInputRequest:
    """A request for human input."""
    id: str
    task_id: str
    question: str
    context: dict[str, Any] = field(default_factory=dict)
    options: list[str] | None = None
    required: bool = True
    timeout_seconds: int = 300
    created_at: datetime = field(default_factory=datetime.now)
    status: HumanInputStatus = HumanInputStatus.PENDING
    response: str | None = None
    responded_at: datetime | None = None


@dataclass
class HumanInputResult:
    """Result of human input."""
    request_id: str
    task_id: str
    status: HumanInputStatus
    response: str | None
    wait_time_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)


class HumanInTheLoop:
    """Manages human-in-the-loop interactions.

    This class handles:
    - Pausing task execution
    - Requesting human input with timeout
    - Resuming execution after input is received
    - Managing multiple concurrent input requests
    """

    def __init__(
        self,
        default_timeout: int = 300,
        auto_proceed_on_timeout: bool = False,
    ):
        """
        Args:
            default_timeout: Default timeout in seconds for input requests
            auto_proceed_on_timeout: Whether to proceed or fail on timeout
        """
        self.default_timeout = default_timeout
        self.auto_proceed_on_timeout = auto_proceed_on_timeout

        self._pending_requests: dict[str, HumanInputRequest] = {}
        self._responses: dict[str, HumanInputResult] = {}
        self._lock = threading.Lock()

        # Default input handler (can be replaced)
        self._input_handler: Callable[[HumanInputRequest], str | None] | None = None

    def set_input_handler(
        self,
        handler: Callable[[HumanInputRequest], str | None]
    ) -> None:
        """Set the handler function for human input.

        Args:
            handler: Function that takes a HumanInputRequest and returns input string
        """
        self._input_handler = handler

    def request_input(
        self,
        task_id: str,
        question: str,
        context: dict[str, Any] | None = None,
        options: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> HumanInputResult:
        """Request human input for a task.

        Args:
            task_id: ID of the task requesting input
            question: Question to ask the human
            context: Additional context for the question
            options: Optional list of valid options
            timeout_seconds: Timeout for this request

        Returns:
            HumanInputResult with the response
        """
        timeout = timeout_seconds or self.default_timeout
        request_id = f"input_{task_id}_{int(time.time())}"

        request = HumanInputRequest(
            id=request_id,
            task_id=task_id,
            question=question,
            context=context or {},
            options=options,
            timeout_seconds=timeout,
        )

        with self._lock:
            self._pending_requests[request_id] = request

        start_time = time.time()

        # If we have an input handler, use it
        if self._input_handler:
            response = self._input_handler(request)
            if response is not None:
                request.status = HumanInputStatus.RECEIVED
                request.response = response
                request.responded_at = datetime.now()

                with self._lock:
                    self._pending_requests[request_id] = request

                return HumanInputResult(
                    request_id=request_id,
                    task_id=task_id,
                    status=HumanInputStatus.RECEIVED,
                    response=response,
                    wait_time_seconds=time.time() - start_time,
                )

        # Otherwise, wait for input with timeout
        while (time.time() - start_time) < timeout:
            with self._lock:
                updated_request = self._pending_requests.get(request_id)

            if updated_request and updated_request.status == HumanInputStatus.RECEIVED:
                return HumanInputResult(
                    request_id=request_id,
                    task_id=task_id,
                    status=HumanInputStatus.RECEIVED,
                    response=updated_request.response,
                    wait_time_seconds=time.time() - start_time,
                )

            # Check for timeout
            if (time.time() - start_time) >= timeout:
                request.status = HumanInputStatus.TIMEOUT
                with self._lock:
                    self._pending_requests[request_id] = request

                return HumanInputResult(
                    request_id=request_id,
                    task_id=task_id,
                    status=HumanInputStatus.TIMEOUT,
                    response=None if self.auto_proceed_on_timeout else None,
                    wait_time_seconds=timeout,
                )

            time.sleep(0.1)

        # Should not reach here, but handle it
        request.status = HumanInputStatus.TIMEOUT
        return HumanInputResult(
            request_id=request_id,
            task_id=task_id,
            status=HumanInputStatus.TIMEOUT,
            response=None,
            wait_time_seconds=timeout,
        )

    def provide_input(
        self,
        request_id: str,
        response: str
    ) -> bool:
        """Provide input for a pending request.

        Args:
            request_id: ID of the input request
            response: The response string

        Returns:
            True if request was found and updated, False otherwise
        """
        with self._lock:
            if request_id in self._pending_requests:
                request = self._pending_requests[request_id]
                request.status = HumanInputStatus.RECEIVED
                request.response = response
                request.responded_at = datetime.now()
                self._pending_requests[request_id] = request
                return True
        return False

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending input request.

        Args:
            request_id: ID of the input request

        Returns:
            True if request was found and cancelled, False otherwise
        """
        with self._lock:
            if request_id in self._pending_requests:
                request = self._pending_requests[request_id]
                request.status = HumanInputStatus.CANCELLED
                self._pending_requests[request_id] = request
                return True
        return False

    def get_pending_requests(self) -> list[HumanInputRequest]:
        """Get all pending input requests."""
        with self._lock:
            return [
                req for req in self._pending_requests.values()
                if req.status == HumanInputStatus.PENDING
            ]

    def get_request_status(self, request_id: str) -> HumanInputStatus | None:
        """Get the status of an input request."""
        with self._lock:
            request = self._pending_requests.get(request_id)
            return request.status if request else None

    def generate_prompt(self, request: HumanInputRequest) -> str:
        """Generate a human-readable prompt for the request.

        Args:
            request: The input request

        Returns:
            Formatted prompt string
        """
        lines = [
            "=" * 60,
            f"HUMAN INPUT REQUIRED",
            "=" * 60,
            "",
            f"Task: {request.task_id}",
            f"Question: {request.question}",
            "",
        ]

        if request.context:
            lines.append("Context:")
            for key, value in request.context.items():
                lines.append(f"  - {key}: {value}")
            lines.append("")

        if request.options:
            lines.append("Options:")
            for i, option in enumerate(request.options, 1):
                lines.append(f"  {i}. {option}")
            lines.append("")

        lines.extend([
            f"Timeout: {request.timeout_seconds} seconds",
            "=" * 60,
        ])

        return "\n".join(lines)


# ============================================================================
# Mock Human Input for Demo
# ============================================================================

class MockHumanInputHandler:
    """Mock handler for human input in demo mode.

    This simulates human input by providing pre-configured responses
    or allowing manual input during demo.
    """

    def __init__(self):
        self._responses: dict[str, str] = {}
        self._manual_mode = False
        self._manual_input: str | None = None
        self._input_event = threading.Event()

    def add_response(self, question_pattern: str, response: str) -> None:
        """Add a canned response for a question pattern.

        Args:
            question_pattern: Pattern to match in the question
            response: Pre-configured response
        """
        self._responses[question_pattern] = response

    def enable_manual_mode(self) -> None:
        """Enable manual input mode (blocks until input provided)."""
        self._manual_mode = True

    def provide_manual_input(self, input_text: str) -> None:
        """Provide manual input (used with manual_mode)."""
        self._manual_input = input_text
        self._input_event.set()

    def get_input(self, request: HumanInputRequest) -> str | None:
        """Get input for a request.

        Args:
            request: The input request

        Returns:
            Input string or None
        """
        # Check for canned response
        for pattern, response in self._responses.items():
            if pattern.lower() in request.question.lower():
                return response

        # If in manual mode, wait for input
        if self._manual_mode:
            print("\n" + "=" * 60)
            print("WAITING FOR MANUAL INPUT")
            print("=" * 60)
            print(f"Question: {request.question}")
            if request.options:
                print("Options:")
                for i, opt in enumerate(request.options, 1):
                    print(f"  {i}. {opt}")
            print("=" * 60)

            self._input_event.wait()
            self._input_event.clear()

            return self._manual_input

        # Default: auto-respond for demo
        return "[Auto-response for demo]"

    def reset(self) -> None:
        """Reset the handler state."""
        self._responses.clear()
        self._manual_mode = False
        self._manual_input = None
        self._input_event.clear()
